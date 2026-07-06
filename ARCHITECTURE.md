# Architecture & Development Documentation

This document explains how the SAM + SigLIP zero-shot visual search pipeline
works internally, the problems found in the original prototype and how they
were fixed, and diagrams of the system's control flow, data flow, and
deployment process.

> Diagrams use [Mermaid](https://mermaid.js.org/), which GitHub, GitLab, and
> most modern Markdown viewers render automatically — no extra tooling
> needed to view them.

---

## 1. System overview

The project answers one question: **"where is `<text query>` in this
image?"** — without any training, fine-tuning, or fixed label set.

It does this by combining two independent, pre-trained models that were
never designed to work together:

| Model | Role | Why it's needed |
|---|---|---|
| **SAM** (`segment-anything`, `vit_b`) | Class-agnostic object proposal | Finds *every* plausible object boundary in the image, with no idea what any of them are |
| **SigLIP** (`google/siglip-base-patch16-224`) | Zero-shot vision-language matching | Scores how well each proposed object matches the free-text query |

Neither model alone can do this task: SAM has no language understanding,
and SigLIP alone has no way to localize objects within an image — it can
only compare whole images to text. Chaining them turns two narrow
capabilities into open-vocabulary object localization.

---

## 2. Pipeline diagram

```mermaid
flowchart TD
    A[Input: image URL + text query] --> B[Download & load image]
    B --> C{Larger than<br/>max-dimension?}
    C -- yes --> D[Downscale image]
    C -- no --> E
    D --> E[Convert to numpy array]
    E --> F[SAM: generate mask proposals]
    F --> G[Extract bounding boxes<br/>+ predicted_iou scores]
    G --> H[Convert to tensors]
    H --> I["NMS filter<br/>(IoU threshold 0.7)"]
    I --> J[Surviving object proposals]
    J --> K[For each proposal:<br/>tight crop]
    J --> L["For each proposal:<br/>context crop (1.3x box)"]
    K --> M["SigLIP scoring<br/>(chunked batches)"]
    L --> N["SigLIP scoring<br/>(chunked batches)"]
    M --> O["Combine: 0.7 * tight_score<br/>+ 0.3 * context_score"]
    N --> O
    O --> P[Argmax: best-matching object]
    P --> Q[Save crop + print score & box]
```

---

## 3. Sequence diagram — running a query

This shows the runtime interaction between the CLI, the two models, and the
filesystem for a single `python main.py --image-url ... --query ...` call.

```mermaid
sequenceDiagram
    actor User
    participant CLI as main.py
    participant IMG as Image loader
    participant SAM as SAM (mask_generator)
    participant NMS as torchvision.ops.nms
    participant SigLIP as SigLIP (model + processor)
    participant FS as Filesystem

    User->>CLI: --image-url, --query, --output
    CLI->>SAM: load_models() [SAM + SigLIP onto device]
    SAM-->>CLI: mask_generator, language_model, language_processor
    CLI->>IMG: load_image(url, max_dimension)
    IMG-->>CLI: PIL image (resized if needed)
    CLI->>SAM: mask_generator.generate(image_np)
    SAM-->>CLI: list of masks (bbox, predicted_iou, ...)
    CLI->>NMS: nms(boxes, scores, iou_threshold=0.7)
    NMS-->>CLI: indices of surviving boxes
    CLI->>CLI: build tight_crops[] and context_crops[]
    loop for each batch of size --batch-size
        CLI->>SigLIP: score tight_crops batch
        SigLIP-->>CLI: batch scores (tight)
        CLI->>SigLIP: score context_crops batch
        SigLIP-->>CLI: batch scores (context)
        CLI->>CLI: torch.cuda.empty_cache()
    end
    CLI->>CLI: final_score = 0.7*tight + 0.3*context
    CLI->>CLI: best_idx = argmax(final_score)
    CLI->>FS: save best crop to --output
    CLI-->>User: print box, score, output path
```

---

## 4. Sequence diagram — Docker build & run

```mermaid
sequenceDiagram
    actor Dev as Developer
    participant Docker as Docker Engine
    participant Build as Build container
    participant PyPI as PyPI / GitHub
    participant FB as fbaipublicfiles.com
    participant Img as sam-siglip-search image
    participant Run as Run container
    participant Host as Host filesystem

    Dev->>Docker: docker build -t sam-siglip-search .
    Docker->>Build: FROM python:3.11-slim
    Build->>Build: apt-get install curl git libgl1 libglib2.0-0
    Build->>PyPI: pip install -r requirements.txt
    PyPI-->>Build: torch, torchvision, transformers,<br/>segment-anything, Pillow, requests, numpy
    Build->>Build: COPY main.py, download_weights.sh
    Build->>FB: download_weights.sh curls SAM checkpoint
    FB-->>Build: sam_vit_b_01ec64.pth (~375MB)
    Build->>Img: layers committed
    Docker-->>Dev: image built (fully self-contained)

    Dev->>Docker: docker run -v ./output:/app/output sam-siglip-search --image-url ... --query ...
    Docker->>Run: start container from Img
    Run->>Run: main.py executes pipeline (see previous diagram)
    Run->>Host: write result crop to mounted /app/output
    Run-->>Dev: stdout: box, score, output path
```

---

## 5. Memory-safety data flow

This is the part of the system most likely to fail on a memory-constrained
machine, and the diagram below shows where each guard rail sits.

```mermaid
flowchart LR
    subgraph Guard1["Guard 1: image size"]
        A1[Raw downloaded image] --> A2{"max(w,h) > 1024?"}
        A2 -- yes --> A3[thumbnail resize]
        A2 -- no --> A4[unchanged]
        A3 --> A5[Bounded-size image]
        A4 --> A5
    end

    subgraph Guard2["Guard 2: proposal density"]
        A5 --> B1["SAM.generate()<br/>points_per_side=16"]
        B1 --> B2[Fewer, coarser masks<br/>vs. SAM default of 32]
    end

    subgraph Guard3["Guard 3: batched scoring"]
        B2 --> C1[NMS-filtered proposals]
        C1 --> C2["Chunk into groups of<br/>batch_size (default 8)"]
        C2 --> C3["SigLIP forward pass<br/>on one chunk"]
        C3 --> C4[torch.cuda.empty_cache]
        C4 --> C2
        C2 --> C5[All chunks scored]
    end

    C5 --> D[torch.cuda.OutOfMemoryError<br/>caught at CLI level]
    D --> E["Actionable error:<br/>suggests --cpu, lower<br/>--batch-size / --points-per-side"]
```

---

## 6. Problems found and solutions

These were identified while turning the original notebook prototype into a
runnable, shareable project. Each row is something that would have failed
on a machine other than the one the notebook happened to be developed on.

| # | Problem | Why it happens | Solution implemented |
|---|---|---|---|
| 1 | `ImportError: No module named 'segment_anything'` on a fresh machine | The install cell (`Part 1`) only ran `pip install ultralytics transformers`, but the code imports `from segment_anything import ...`. `ultralytics` does not provide that module. | `requirements.txt` installs `segment-anything` directly from its GitHub source; `Dockerfile` also installs `git`, which pip needs to fetch it. |
| 2 | `FileNotFoundError` / silent failure loading `sam_vit_b_01ec64.pth` | The notebook assumes the checkpoint file already exists locally; no download step exists anywhere in the notebook. | Added `download_weights.sh`, which fetches the official checkpoint from Meta's public bucket. The `Dockerfile` runs this at build time so the image is self-contained and works fully offline afterward. |
| 3 | Notebook's install cell fails outright in network-isolated environments (e.g., Kaggle with internet disabled), but the notebook prints "✅ All libraries are ready" regardless | The `pip install` output is never checked; success is printed unconditionally. | The CLI (`main.py`) fails loudly with a real exception and exit code if models can't load, instead of a false-positive success message. |
| 4 | Potential CUDA out-of-memory during the SigLIP scoring step in crowded images | The original code builds `tight_crops` and `context_crops` lists of *all* NMS-surviving objects and sends each list through SigLIP in a **single** batched forward pass. A busy scene can yield many dozens of proposals, and this batch size is unbounded — the one place memory usage scales with image content, which the "resize to 1024px" step doesn't protect against. | Added `_score_crops_in_batches()`, which processes crops in fixed-size chunks (`--batch-size`, default 8), clearing the CUDA cache between chunks. Memory use for this stage is now roughly constant regardless of how many objects SAM finds. |
| 5 | No way to run on CPU-only or memory-constrained machines without editing code | All parameters (image size cap, SAM density, device) were hardcoded in notebook cells. | Exposed `--cpu`, `--max-dimension`, `--points-per-side`, and `--batch-size` as CLI flags so memory/quality tradeoffs can be tuned per machine without touching source. |
| 6 | Full-resolution numpy array and raw SAM mask segmentation arrays kept in memory longer than needed | The notebook keeps every intermediate object alive for the rest of the cell's execution. | `main.py` explicitly `del`s the mask list and the full-resolution numpy array as soon as their data has been extracted, followed by `gc.collect()`, before the more memory-hungry SigLIP stage runs. |
| 7 | No graceful handling of an actual OOM if it still occurs | The notebook's only error handling was a blanket `except Exception` around the entire pipeline, printing a generic message. | `main.py`'s CLI entrypoint specifically catches `torch.cuda.OutOfMemoryError` and prints actionable next steps (lower batch size / points-per-side / max-dimension, or add `--cpu`). |
| 8 | Not portable — required manually re-running install/model-loading cells per session, on a specific host with pre-placed files | Notebook-only workflow, tied to one interactive Kaggle session. | Repackaged as a CLI script + `Dockerfile`, so the exact same environment (Python version, dependency versions, model weights) is reproducible on any machine with Docker installed. |

---

## 7. Design notes / tuning knobs

- **`tight_weight` / `context_weight` (0.7 / 0.3 by default)** — controls how
  much the surrounding context of an object influences its score, versus
  the object alone. Passed as arguments to `find_best_match()` if you want
  to experiment.
- **`context_scale` (1.3x by default)** — how much wider the "context crop"
  is than the tight bounding box.
- **`points_per_side` (16 by default, SAM's own default is 32)** — directly
  trades proposal recall for speed/memory. Raise it for small/cluttered
  objects that might be missed; lower it further for very constrained
  hardware.
- **`iou_threshold` (0.7, fixed in `propose_objects`)** — NMS aggressiveness
  for removing duplicate/overlapping proposals.

## 8. Known limitations

- Zero-shot performance depends on how well SigLIP's training data covers
  the query's vocabulary and visual concept — very unusual or abstract
  queries will score poorly across all candidates.
- SAM can miss very small, heavily occluded, or low-contrast objects
  regardless of the query.
- This is a single-best-match pipeline — it doesn't currently return ranked
  top-k matches, though the `final_scores` tensor computed internally makes
  that a straightforward extension.
