FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# ---- OS-level dependencies ----
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libopenblas-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Conda Environment ----
COPY environment.yml /tmp/environment.yml
RUN conda update -n base -c defaults conda && \
    conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate your custom env by default
SHELL ["conda", "run", "-n", "pytorch3d", "/bin/bash", "-c"]

WORKDIR /workspace
CMD ["python", "./LiDiff/lidiff/train.py", "--config", "./LiDiff/lidiff/config/config.yaml"]
