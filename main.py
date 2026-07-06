#!/usr/bin/env python3
"""
SAM + SigLIP Zero-Shot Visual Search
=====================================
Pipeline:
  1. Segment Anything (SAM, vit_b) proposes object masks/boxes for an image.
  2. Boxes are filtered with Non-Maximum Suppression (NMS) to remove duplicates.
  3. Each surviving object is cropped two ways: a tight crop and a wider
     "context" crop (1.3x the box size).
  4. Google's SigLIP scores every crop against a text query (zero-shot).
  5. Tight-crop and context-crop scores are combined (0.7 / 0.3 weighting)
     and the highest-scoring object is returned as the best match.

Usage:
    python main.py --image-url "https://images.pexels.com/photos/235294/pexels-photo-235294.jpeg" \
                    --query "a basket of red apples" \
                    --output result.jpg
"""

import argparse
import gc
import sys
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from torchvision.ops import nms

# ---------------------------------------------------------------------------
# Configuration defaults (safe for machines with limited GPU/CPU memory)
# ---------------------------------------------------------------------------
DEFAULT_MAX_DIMENSION = 1024        # resize any image larger than this (px)
DEFAULT_POINTS_PER_SIDE = 16        # lower = fewer SAM masks = less memory
DEFAULT_SIGLIP_BATCH_SIZE = 8       # chunk size for SigLIP forward passes
SAM_CHECKPOINT_PATH = "weights/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"


def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models(device: str, points_per_side: int = DEFAULT_POINTS_PER_SIDE):
    """Load SAM (mask generator) and SigLIP (vision-language model)."""
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from transformers import SiglipModel, SiglipProcessor

    print(f"Loading models onto device: {device}")

    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA requested but not available.")

    # --- SAM ---
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )
    print("SAM model loaded.")

    # --- SigLIP ---
    language_model = SiglipModel.from_pretrained(SIGLIP_MODEL_NAME).to(device)
    language_model.eval()
    language_processor = SiglipProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    print("SigLIP model loaded.")

    return mask_generator, language_model, language_processor


def load_image(image_url: str, max_dimension: int = DEFAULT_MAX_DIMENSION) -> Image.Image:
    """Download an image and downscale it if it's larger than max_dimension."""
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image_rgb = Image.open(BytesIO(response.content)).convert("RGB")
    print(f"Image loaded. Original size: {image_rgb.size}")

    if max(image_rgb.size) > max_dimension:
        image_rgb.thumbnail((max_dimension, max_dimension))
        print(f"Resized image to: {image_rgb.size}")

    return image_rgb


def propose_objects(mask_generator, image_np: np.ndarray, iou_threshold: float = 0.7):
    """Run SAM, then filter overlapping boxes with NMS. Returns list of
    {'box': [x1,y1,x2,y2]} proposals (no crops yet, to save memory)."""
    masks = mask_generator.generate(image_np)
    print(f"Found {len(masks)} potential object masks.")

    if not masks:
        return []

    boxes_for_nms = []
    scores_for_nms = []
    for mask in masks:
        bbox = mask["bbox"]
        box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        boxes_for_nms.append(box)
        scores_for_nms.append(mask["predicted_iou"])

    # masks can be large (segmentation arrays); drop the reference once we
    # have what we need so it can be garbage collected before the next stage
    del masks
    gc.collect()

    boxes_tensor = torch.tensor(boxes_for_nms, dtype=torch.float32)
    scores_tensor = torch.tensor(scores_for_nms, dtype=torch.float32)
    nms_indices = nms(boxes_tensor, scores_tensor, iou_threshold=iou_threshold)

    proposals = [{"box": boxes_for_nms[i.item()]} for i in nms_indices]
    print(f"Kept {len(proposals)} unique objects after NMS.")
    return proposals


def _score_crops_in_batches(language_model, language_processor, device, text_query, crops, batch_size):
    """Run SigLIP over a list of PIL crops in small batches to bound peak
    memory use, regardless of how many objects SAM found."""
    all_scores = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            inputs = language_processor(
                text=[text_query] * len(batch),
                images=batch,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            outputs = language_model(**inputs)
            batch_scores = outputs.logits_per_image.diag()
            all_scores.append(batch_scores.detach().cpu())

            del inputs, outputs
            if device == "cuda":
                torch.cuda.empty_cache()

    return torch.cat(all_scores)


def find_best_match(
    image_rgb: Image.Image,
    proposals,
    language_model,
    language_processor,
    device: str,
    text_query: str,
    context_scale: float = 1.3,
    tight_weight: float = 0.7,
    context_weight: float = 0.3,
    batch_size: int = DEFAULT_SIGLIP_BATCH_SIZE,
):
    """Score each proposal's tight crop and wider context crop against the
    text query, then combine with a weighted average."""
    if not proposals:
        return None

    w, h = image_rgb.size
    tight_crops = []
    context_crops = []

    for proposal in proposals:
        box = proposal["box"]
        tight_crops.append(image_rgb.crop(box))

        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        bw = (box[2] - box[0]) * context_scale
        bh = (box[3] - box[1]) * context_scale
        context_box = [
            max(0, cx - bw / 2),
            max(0, cy - bh / 2),
            min(w, cx + bw / 2),
            min(h, cy + bh / 2),
        ]
        context_crops.append(image_rgb.crop(context_box))

    scores_tight = _score_crops_in_batches(
        language_model, language_processor, device, text_query, tight_crops, batch_size
    )
    scores_context = _score_crops_in_batches(
        language_model, language_processor, device, text_query, context_crops, batch_size
    )

    final_scores = (tight_weight * scores_tight) + (context_weight * scores_context)
    best_idx = int(torch.argmax(final_scores).item())
    best_score = float(final_scores[best_idx].item())

    return {
        "box": proposals[best_idx]["box"],
        "score": best_score,
        "crop": tight_crops[best_idx],
    }


def run_pipeline(
    image_url: str,
    text_query: str,
    output_path: str,
    device: str,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    points_per_side: int = DEFAULT_POINTS_PER_SIDE,
    batch_size: int = DEFAULT_SIGLIP_BATCH_SIZE,
):
    mask_generator, language_model, language_processor = load_models(device, points_per_side)

    print(f"\nProcessing image: {image_url}")
    print(f"Searching for: '{text_query}'")

    image_rgb = load_image(image_url, max_dimension)
    image_np = np.array(image_rgb)

    print("Step A: Finding potential objects with SAM...")
    proposals = propose_objects(mask_generator, image_np)

    # Free the full-resolution numpy array; we only need PIL crops from here.
    del image_np
    gc.collect()

    if not proposals:
        print("No unique objects were found after filtering.")
        return None

    print("Step B: Scoring objects against text query with SigLIP...")
    result = find_best_match(
        image_rgb,
        proposals,
        language_model,
        language_processor,
        device,
        text_query,
        batch_size=batch_size,
    )

    if result is None:
        print("No match found.")
        return None

    print("\n--- RESULTS ---")
    print(f"Best match found for '{text_query}'")
    print(f"Combined SigLIP score: {result['score']:.4f}")
    print(f"Bounding box (x1, y1, x2, y2): {result['box']}")

    result["crop"].save(output_path)
    print(f"Saved best matching crop to: {output_path}")

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAM + SigLIP zero-shot visual search")
    parser.add_argument("--image-url", required=True, help="URL of the image to search")
    parser.add_argument("--query", required=True, help="Text description of the object to find")
    parser.add_argument("--output", default="result.jpg", help="Path to save the best-match crop")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available")
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=DEFAULT_MAX_DIMENSION,
        help="Resize images larger than this (px) before running SAM",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=DEFAULT_POINTS_PER_SIDE,
        help="SAM mask-generator density; lower uses less memory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_SIGLIP_BATCH_SIZE,
        help="Number of crops scored per SigLIP forward pass",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    device = get_device(force_cpu=args.cpu)

    try:
        run_pipeline(
            image_url=args.image_url,
            text_query=args.query,
            output_path=args.output,
            device=device,
            max_dimension=args.max_dimension,
            points_per_side=args.points_per_side,
            batch_size=args.batch_size,
        )
    except torch.cuda.OutOfMemoryError:
        print(
            "\nGPU ran out of memory. Try re-running with --cpu, a lower "
            "--points-per-side, a smaller --batch-size, or a smaller --max-dimension.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error boundary
        print(f"An error occurred: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
