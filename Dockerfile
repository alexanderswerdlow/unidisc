# Base image with CUDA 12.6.3 and cuDNN
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    SYSTEM=spaces \
    AM_I_IN_A_DOCKER_CONTAINER=Yes \
    PYTHONPATH=/home/appuser/app \
    HF_HOME=/home/appuser/.cache \
    TORCH_HOME=/home/appuser/.cache \
    TMP_DIR=/home/appuser/tmp \
    TRANSFORMERS_CACHE=/home/appuser/.cache/transformers \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies and set Python 3.10 as default
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    git \
    openssh-client \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install `uv`
RUN pip install --upgrade pip \
    && pip install uv

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /home/appuser/app

# Copy dependency files and install dependencies
COPY --chown=appuser pyproject.toml uv.lock README.md ./ 
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN --mount=type=ssh uv sync --no-group dev
RUN --mount=type=ssh uv sync --frozen --no-cache \
    && chown -R appuser:appuser /home/appuser/app/.venv \
    && rm -rf /root/.cache /home/appuser/.cache

# Ensure non-root user has write access to cache and tmp directories
RUN mkdir -p /home/appuser/.cache/transformers /home/appuser/tmp /home/appuser/.cache \
    && chown -R appuser:appuser /home/appuser/.cache /home/appuser/tmp/ /home/appuser/app/

RUN chmod -R 777 /tmp

# Copy application code
COPY --chown=appuser demo demo
COPY --chown=appuser unidisc unidisc
COPY --chown=appuser models models
COPY --chown=appuser configs configs
COPY --chown=appuser third_party third_party
COPY --chown=appuser ckpts ckpts
COPY --chown=appuser ./__* ./
COPY --chown=appuser ./*.py ./
COPY --chown=appuser ./archive/pytorch_model_fsdp.bin ./

# Switch to non-root user
USER appuser

# Expose port for Gradio
EXPOSE 5003

# Command to run the application
CMD ["bash", "demo/demo.sh"]

# DOCKER_BUILDKIT=1 docker build --ssh default --network=host -t unidisc .
# docker run --network=host -it -p 5003:5003 unidisc