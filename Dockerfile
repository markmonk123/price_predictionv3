# syntax=docker/dockerfile:1.7
#
# Dockerfile for priceprediction.py
#   - Base: Ubuntu 22.04 (matches the project's WSL dev environment and the
#     cuML 26.4.0 wheel matrix pinned in requirements.txt).
#   - Python: 3.11 (the WSL environment used by sync-to-wsl.ps1 runs 3.11,
#     and requirements.txt declares python_version < "3.13").
#   - pip, setuptools, and npm are forced to the latest version via
#     `--no-cache-dir` / `-U` so Docker layers cannot reuse a stale wheel
#     or a baked-in npm tarball.
#
# Build:
#   docker build -t priceprediction:latest .
#
# Run (GPU / RAPIDS / cuML path):
#   docker run --gpus all --rm \
#     -v "${PWD}:/app" -w /app \
#     priceprediction:latest
#
# Run (CPU only, ignoring the CUDA extra-index in requirements.txt):
#   docker run --rm priceprediction:latest
#

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

# ----------------------------------------------------------------------------
# Stage 1 — Ubuntu 22.04 + system toolchain + Node.js
# ----------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_AUDIT=false \
    NPM_CONFIG_CACHE=/tmp/.npm-cache

# System packages: build tools (needed by some pip wheels), runtime libs
# for numpy/scipy (libopenblas, libomp), TLS / XML libs, git/curl, and the
# Node.js 20.x toolchain so we can refresh npm to a non-cached version.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
        wget \
        build-essential \
        gcc \
        g++ \
        make \
        libopenblas-dev \
        libomp-dev \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        software-properties-common \
 && mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Stage 2 — Python 3.11 built from source.
# Ubuntu 22.04 ships 3.10; we want 3.11 to match requirements.txt and the
# WSL dev environment referenced by sync-to-wsl.ps1.
# ----------------------------------------------------------------------------
ARG PYTHON_VERSION

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libnccl2 libnccl-dev \
        libnvinfer8 libnvonnxparsers8 libnvparsers8 \
 && cd /usr/local/src \
 && wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}.12/Python-${PYTHON_VERSION}.12.tar.xz" \
 && tar -xf "Python-${PYTHON_VERSION}.12.tar.xz" \
 && cd "Python-${PYTHON_VERSION}.12" \
 && ./configure --enable-optimizations --with-ensurepip=upgrade --enable-shared \
 && make -j"$(nproc)" \
 && make altinstall \
 && ldconfig \
 && cd /usr/local/src \
 && rm -rf "Python-${PYTHON_VERSION}.12" "Python-${PYTHON_VERSION}.12.tar.xz" \
 && rm -rf /var/lib/apt/lists/*

# Make python3 / pip point at the freshly built 3.11.
RUN update-alternatives --install /usr/local/bin/python3 python3 /usr/local/bin/python3.11 100 \
 && update-alternatives --install /usr/local/bin/python  python  /usr/local/bin/python3.11 100 \
 && hash -r

# ----------------------------------------------------------------------------
# Stage 3 — refresh pip, setuptools, and npm to the *latest, non-cached* version.
#   * pip / setuptools: forced via `--no-cache-dir --upgrade` so Docker layers
#     always fetch a fresh wheel from PyPI (no BuildKit / host cache shortcut).
#   * npm: forced via `npm install -g npm@latest` with the npm cache redirected
#     to /tmp/.npm-cache (set in ENV) so the runner can't reuse a stale tarball.
# ----------------------------------------------------------------------------
RUN python -m pip install --no-cache-dir --upgrade "pip>=23" \
 && python -m pip install --no-cache-dir --upgrade "setuptools>=68" \
 && python -m pip install --no-cache-dir --upgrade "wheel" \
 && rm -rf /root/.cache/pip

RUN npm install -g npm@latest \
 && npm cache clean --force

# Sanity: prove the freshly fetched toolchain made it in.
RUN python --version \
 && pip --version \
 && setuptools --version \
 && npm --version

# ----------------------------------------------------------------------------
# Stage 4 — runtime dependencies and the application.
# ----------------------------------------------------------------------------
WORKDIR /app

# Copy only the metadata first so the pip install layer is cached across
# source-only code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && rm -rf /root/.cache/pip

# Bring in the rest of the source.
COPY . .

# Keep the entrypoint script executable.
RUN chmod +x ./priceprediction.py 2>/dev/null || true

# Default: run the project's standard invocation. Override with:
#   docker run ... priceprediction:latest <your-command>
CMD ["python", "priceprediction.py"]
