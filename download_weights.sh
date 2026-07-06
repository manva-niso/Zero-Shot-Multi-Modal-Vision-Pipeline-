#!/usr/bin/env bash
# Downloads the SAM ViT-B checkpoint used by main.py, if it isn't already
# present. Safe to run multiple times.
set -euo pipefail

WEIGHTS_DIR="$(dirname "$0")/weights"
CHECKPOINT_PATH="${WEIGHTS_DIR}/sam_vit_b_01ec64.pth"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

mkdir -p "${WEIGHTS_DIR}"

if [ -f "${CHECKPOINT_PATH}" ]; then
    echo "SAM checkpoint already present at ${CHECKPOINT_PATH}, skipping download."
else
    echo "Downloading SAM ViT-B checkpoint (~375MB)..."
    curl -L --fail -o "${CHECKPOINT_PATH}" "${CHECKPOINT_URL}"
    echo "Saved to ${CHECKPOINT_PATH}"
fi
