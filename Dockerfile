# Karpathian proof-test container
#
# This is the official miner container that runs the canonical training
# procedure for a submitted patch and produces the attested proof bundle.
#
# In Phase 0.5+ its image digest = the "container measurement" committed
# on-chain. No other workload can produce a valid attestation against this
# measurement. The validator rejects any submission whose attested measurement
# doesn't match.
#
# Build:
#   docker build -t karpathian-proof:latest .
#
# Run a proof test:
#   docker run --gpus all --rm \
#     -v /path/to/data:/data:ro \
#     -v /path/to/submission:/submission:ro \
#     -v /path/to/output:/output \
#     karpathian-proof:latest \
#       --submission /submission --out-dir /output
#
# For reproducible builds:
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg BUILDKIT_INLINE_CACHE=1 \
#     -t karpathian-proof:$(git rev-parse --short HEAD) .
#
# The image digest (sha256) is the container measurement:
#   docker inspect --format='{{.RepoDigests}}' karpathian-proof:latest

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    patch \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies first for layer caching.
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -e /app

# Copy the protocol source — model, recipe, data, eval, calibration, proof.
# This is the "canonical training code" whose hash is the measurement.
COPY model/ /app/model/
COPY recipe/ /app/recipe/
COPY data/ /app/data/
COPY eval/ /app/eval/
COPY calibration/ /app/calibration/
COPY proof/ /app/proof/
COPY miner/ /app/miner/
COPY validator/ /app/validator/
COPY configs/ /app/configs/
COPY restricted_files.yaml /app/restricted_files.yaml

WORKDIR /app

# Verify the model code loads.
RUN python -c "from model import KarpathianBase, KarpathianConfig; print('model import ok')"
RUN python -c "from proof.runner import run_proof_test; print('proof runner import ok')"

# The entry point is the proof runner.
# Mounts expected at runtime:
#   /data       (ro)  training data shards + manifest
#   /submission (ro)  patch.diff + proof_request.json
#   /output           where the proof bundle is written
ENTRYPOINT ["python", "-m", "proof.runner"]
CMD ["--help"]

# --- Labels for traceability ---
ARG GIT_SHA="unknown"
LABEL org.opencontainers.image.source="https://github.com/KarpathianBase/karpathian"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.description="Karpathian proof-test container — canonical training + attestation"
