# SAM + SigLIP Zero-Shot Visual Search

Find the object in an image that best matches a free-text description —
with no training or labeled dataset required.

Given an image and a text query like `"a basket of red apples"`, this
pipeline automatically proposes every object in the scene, filters out
duplicates, and returns the crop that best matches the query, along with a
confidence score.

## How it works

```
 image ──▶ SAM (segment-anything) ──▶ object mask proposals
                                            │
                                            ▼
                              filter overlaps with NMS (IoU 0.7)
                                            │
                                            ▼
                for each surviving object: take a tight crop
                and a wider "context" crop (1.3x the box size)
                                            │
                                            ▼
                     SigLIP scores every crop against the
                        text query (zero-shot, batched)
                                            │
                                            ▼
        final score = 0.7 * tight_crop_score + 0.3 * context_crop_score
                                            │
                                            ▼
                    highest-scoring object = the answer
```

**Models used**
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything), `vit_b` checkpoint — proposes candidate object masks/boxes without needing to know what it's looking for in advance.
- [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) (`google/siglip-base-patch16-224`) — a vision-language model used to score how well each cropped object matches the text query.

Combining a class-agnostic segmenter with a vision-language matcher lets the
pipeline find *any* object described in plain English, not just objects from
a fixed label set.

## Project structure

```
.
├── main.py               # the full pipeline, as a CLI script
├── requirements.txt       # Python dependencies
├── download_weights.sh    # fetches the SAM checkpoint
├── Dockerfile             # builds a self-contained, runnable image
├── .dockerignore
├── .gitignore
└── weights/               # SAM checkpoint lives here (downloaded, not committed)
```

## Quick start (Docker — recommended)

Docker is the easiest way to run this on any machine, since it bundles
Python, all dependencies, and the ~375MB SAM model weights into one image.
You do **not** need Python, PyTorch, or CUDA installed on your host machine.

```bash
# 1. Build the image (downloads the SAM checkpoint during build, ~1-3 min)
docker build -t sam-siglip-search .

# 2. Run it, mounting a local folder to receive the output image
mkdir -p output
docker run --rm -v "$(pwd)/output:/app/output" sam-siglip-search \
  --image-url "https://images.pexels.com/photos/235294/pexels-photo-235294.jpeg" \
  --query "a basket of red apples" \
  --output /app/output/result.jpg
```

The best-matching crop will appear at `output/result.jpg`, and the console
will print the bounding box and confidence score.

### Running with a GPU

If your machine has an NVIDIA GPU, the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed, and a CUDA-compatible base image, add `--gpus all`:

```bash
docker run --rm --gpus all -v "$(pwd)/output:/app/output" sam-siglip-search \
  --image-url "<your image url>" \
  --query "<your query>" \
  --output /app/output/result.jpg
```

The script auto-detects CUDA and falls back to CPU automatically — no code
changes needed either way.

## Running without Docker

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
./download_weights.sh           # downloads weights/sam_vit_b_01ec64.pth

python main.py \
  --image-url "https://images.pexels.com/photos/235294/pexels-photo-235294.jpeg" \
  --query "a basket of red apples" \
  --output result.jpg
```

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--image-url` | *(required)* | URL of the image to search |
| `--query` | *(required)* | Text description of the object to find |
| `--output` | `result.jpg` | Path to save the best-matching crop |
| `--cpu` | off | Force CPU even if a GPU is available |
| `--max-dimension` | `1024` | Images larger than this (px) are downscaled before SAM runs |
| `--points-per-side` | `16` | SAM mask density; lower = fewer proposals = less memory/time |
| `--batch-size` | `8` | Number of crops scored per SigLIP forward pass |

## Memory safety

Large or crowded images can generate a lot of object proposals, and pushing
too much data through a model at once is the most common cause of
out-of-memory (OOM) crashes. This project guards against that in three
places:

1. **Input image downscaling** — any image wider or taller than
   `--max-dimension` (1024px by default) is shrunk before SAM ever sees it.
2. **Conservative SAM sampling** — `--points-per-side` defaults to 16
   (instead of SAM's default of 32), which finds fewer, coarser proposals
   and uses substantially less memory and compute.
3. **Chunked SigLIP scoring** — instead of sending every cropped object
   through SigLIP in a single batch (the previous behavior, which could
   scale unpredictably with how many objects SAM finds in a crowded scene),
   crops are scored in fixed-size batches (`--batch-size`, default 8) with
   the GPU cache cleared between batches.

If you still hit a CUDA out-of-memory error, re-run with `--cpu`, or lower
`--points-per-side`, `--batch-size`, and/or `--max-dimension`.

## Notes on accuracy vs. the original prototype

This is a zero-shot heuristic pipeline, not a trained detector, so results
depend on:
- How distinctive the text query is (very generic queries like "an object"
  will match poorly).
- How cleanly SAM segments the object of interest (occluded or very small
  objects may be missed).
- The 0.7 / 0.3 tight/context weighting, which can be tuned in code
  (`find_best_match`'s `tight_weight` / `context_weight` args) for your use
  case.

## License

This project glues together two open-source models. Check their respective
licenses before commercial use:
- Segment Anything: Apache 2.0
- SigLIP / Transformers: Apache 2.0 (model weights license may vary — see
  the model card on Hugging Face)
