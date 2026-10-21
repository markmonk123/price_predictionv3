FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU runtime dependencies. The repository requirements file also
# contains Linux CUDA/cuML packages, which are not needed for this image.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "numpy>=1.26,<3" \
        "pandas>=2.2,<3" \
        "scikit-learn>=1.5,<1.10" \
        "scipy>=1.14,<2" \
        "requests>=2.31,<3" \
        "joblib>=1.3,<2" \
        "python-dateutil>=2.8,<3"

COPY config ./config
COPY priceprediction.py enhanced_forecasting.py enhanced_prediction.py .\
     continuous_training.py ./

CMD ["python", "priceprediction.py"]
