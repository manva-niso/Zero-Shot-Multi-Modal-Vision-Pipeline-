# SAM + SigLIP Zero-Shot Visual Search
# Runs on CPU by default; will automatically use a GPU if the container is
# started with `--gpus all` and a CUDA-capable host + nvidia-container-toolkit.
FROM python:3.11-slim

WORKDIR /app

# System dependencies:
# - curl: to download the SAM checkpoint at build time
# - git: required to pip-install segment-anything from GitHub
# - libgl1, libglib2.0-0: required by Pillow/opencv-related image codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py download_weights.sh ./
RUN chmod +x download_weights.sh

# Bake the SAM checkpoint into the image so the container works fully
# offline after being built (no internet needed at run time).
RUN ./download_weights.sh

# Where results are written; mount a host folder here to retrieve output,
# e.g. `docker run -v $(pwd)/output:/app/output ...`
RUN mkdir -p /app/output

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
