"""
FastAPI Model Serving Endpoint

This module provides a production-ready REST API for serving ML models:
- /health: Health check endpoint
- /predict: Prediction endpoint with input validation

Security features:
- Pydantic input validation
- Request size limits
- CORS configuration via environment variables
- Safe model loading from validated paths
- Structured logging

Usage:
    uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MAX_FEATURES = int(os.getenv("MAX_FEATURES", "10000"))
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "10000"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="Serve imbalanced-learn ensemble models for predictions",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global model storage
MODELS: Dict[str, Any] = {}


class PredictInput(BaseModel):
    """
    Input schema for prediction requests.
    
    Security notes:
    - Validates feature count and sample count limits
    - Ensures numeric dtypes
    - Prevents memory exhaustion attacks
    """
    features: List[List[float]] = Field(
        ...,
        description="2D array of features for prediction",
        example=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    model_name: Optional[str] = Field(
        default="ensemble_balanced_stacking",
        description="Name of the model to use for prediction"
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate feature dimensions and types."""
        if not v:
            raise ValueError("Features cannot be empty")
        
        n_samples = len(v)
        if n_samples > MAX_SAMPLES:
            raise ValueError(
                f"Too many samples: {n_samples} > {MAX_SAMPLES}. "
                "Request size limit exceeded."
            )
        
        # Check that all rows have same length
        n_features = len(v[0]) if v else 0
        for i, row in enumerate(v):
            if len(row) != n_features:
                raise ValueError(
                    f"Inconsistent feature dimensions at row {i}: "
                    f"expected {n_features}, got {len(row)}"
                )
            
            # Validate all features are numeric
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        f"Non-numeric value at row {i}, col {j}: {val}"
                    )
        
        if n_features > MAX_FEATURES:
            raise ValueError(
                f"Too many features: {n_features} > {MAX_FEATURES}. "
                "Request size limit exceeded."
            )
        
        return v
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name to prevent path traversal."""
        if not v:
            return "ensemble_balanced_stacking"
        
        # Prevent path traversal
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Invalid model name: path traversal detected")
        
        # Whitelist allowed model names
        allowed_models = {
            "ensemble_balanced_stacking",
            "ensemble_voting",
            "ensemble_hybrid"
        }
        
        if v not in allowed_models:
            raise ValueError(
                f"Unknown model: {v}. "
                f"Allowed models: {', '.join(allowed_models)}"
            )
        
        return v


class PredictOutput(BaseModel):
    """Output schema for prediction responses."""
    predictions: List[int] = Field(
        ...,
        description="Predicted class labels"
    )
    probabilities: Optional[List[List[float]]] = Field(
        None,
        description="Prediction probabilities (if available)"
    )
    model_used: str = Field(
        ...,
        description="Name of the model used for prediction"
    )


class HealthResponse(BaseModel):
    """Output schema for health check."""
    status: str
    models_loaded: List[str]
    model_count: int


def load_models():
    """
    Load all available models from the models directory.
    
    Security notes:
    - Only loads from specified MODEL_DIR
    - Validates file extensions
    - Catches and logs errors without exposing details
    """
    model_dir = Path(MODEL_DIR)
    
    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        logger.warning("No models will be loaded. Create models/ and add .joblib files.")
        return
    
    logger.info(f"Loading models from {model_dir}...")
    
    # Expected model files
    model_files = [
        "ensemble_balanced_stacking.joblib",
        "ensemble_voting.joblib",
        "ensemble_hybrid.joblib"
    ]
    
    for model_file in model_files:
        model_path = model_dir / model_file
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        try:
            # Load with joblib (safer than pickle)
            model = joblib.load(model_path)
            model_name = model_path.stem
            MODELS[model_name] = model
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_path}: {e}")
    
    if not MODELS:
        logger.warning("No models loaded! Predictions will fail.")
    else:
        logger.info(f"Successfully loaded {len(MODELS)} models")


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting FastAPI model server...")
    load_models()
    logger.info("Server ready!")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status and list of loaded models
    """
    return HealthResponse(
        status="healthy" if MODELS else "no_models",
        models_loaded=list(MODELS.keys()),
        model_count=len(MODELS)
    )


@app.post("/predict", response_model=PredictOutput)
async def predict(input_data: PredictInput):
    """
    Prediction endpoint.
    
    Args:
        input_data: Validated prediction input
        
    Returns:
        Predictions and probabilities
        
    Raises:
        HTTPException: If model not found or prediction fails
        
    Security notes:
    - Input validated by Pydantic
    - Size limits enforced
    - Errors sanitized before returning
    """
    model_name = input_data.model_name
    
    # Check if model is loaded
    if model_name not in MODELS:
        available = list(MODELS.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not loaded. Available: {available}"
        )
    
    model = MODELS[model_name]
    
    try:
        # Convert to numpy array
        X = np.array(input_data.features, dtype=np.float64)
        
        logger.info(
            f"Prediction request: model={model_name}, "
            f"samples={X.shape[0]}, features={X.shape[1]}"
        )
        
        # Make predictions
        predictions = model.predict(X)
        
        # Try to get probabilities (not all models support this)
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X).tolist()
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
        
        return PredictOutput(
            predictions=predictions.tolist(),
            probabilities=probabilities,
            model_used=model_name
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        # Don't expose internal error details
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs for details."
        )


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """
    Middleware to limit request body size.
    
    Security note: Prevents memory exhaustion attacks
    """
    max_size = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB default
    
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length:
            if int(content_length) > max_size:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large"}
                )
    
    response = await call_next(request)
    return response


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=LOG_LEVEL.lower()
    )
