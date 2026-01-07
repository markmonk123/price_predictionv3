"""
FastAPI Model Serving Server

This module provides a REST API for serving trained ML models with:
- Model loading from configurable directory
- Health check endpoint
- Prediction endpoint with input validation
- Request size limiting
- CORS configuration
- Security best practices (filename validation, error handling)
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(',')
MAX_REQUEST_SIZE = int(os.getenv('MAX_REQUEST_SIZE', '1048576'))  # 1MB default

# Allowed model filenames (security: prevent directory traversal)
ALLOWED_MODEL_NAMES = [
    'ensemble_balanced_stacking.joblib',
    'ensemble_voting.joblib',
    'ensemble_hybrid.joblib'
]

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="Serve trained ensemble models for price prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
loaded_models: Dict[str, Any] = {}


@app.middleware("http")
async def check_request_size(request: Request, call_next):
    """Middleware to limit request body size."""
    content_length = request.headers.get('content-length')
    if content_length:
        content_length = int(content_length)
        if content_length > MAX_REQUEST_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request too large. Max size: {MAX_REQUEST_SIZE} bytes"}
            )
    
    response = await call_next(request)
    return response


def validate_model_filename(filename: str) -> bool:
    """
    Validate model filename to prevent directory traversal attacks.
    
    Args:
        filename: The model filename to validate
        
    Returns:
        bool: True if filename is valid and in allowed list
    """
    # Check if filename contains path separators
    if '/' in filename or '\\' in filename or '..' in filename:
        return False
    
    # Check if filename is in allowed list
    return filename in ALLOWED_MODEL_NAMES


def load_models_on_startup():
    """
    Load all available models from MODEL_DIR on server startup.
    
    Only loads models with filenames in ALLOWED_MODEL_NAMES.
    """
    model_path = Path(MODEL_DIR)
    
    if not model_path.exists():
        logger.warning(f"Model directory does not exist: {model_path}")
        return
    
    logger.info(f"Loading models from: {model_path.absolute()}")
    
    for model_file in ALLOWED_MODEL_NAMES:
        model_file_path = model_path / model_file
        
        if model_file_path.exists():
            try:
                model = joblib.load(model_file_path)
                model_name = model_file.replace('.joblib', '')
                loaded_models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_file}: {e}")
        else:
            logger.debug(f"Model file not found: {model_file}")
    
    logger.info(f"Total models loaded: {len(loaded_models)}")


@app.on_event("startup")
async def startup_event():
    """FastAPI startup event handler."""
    logger.info("Starting FastAPI Model Serving Server")
    logger.info(f"Configuration:")
    logger.info(f"  MODEL_DIR: {MODEL_DIR}")
    logger.info(f"  ALLOWED_ORIGINS: {ALLOWED_ORIGINS}")
    logger.info(f"  MAX_REQUEST_SIZE: {MAX_REQUEST_SIZE} bytes")
    
    load_models_on_startup()


# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    model_name: str = Field(..., description="Name of the model to use for prediction")
    features: List[List[float]] = Field(..., description="Feature matrix as list of lists")
    
    @validator('features')
    def validate_features(cls, v):
        """Validate features are not empty."""
        if not v or not v[0]:
            raise ValueError("Features cannot be empty")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "ensemble_balanced_stacking",
                "features": [[1.0, 2.0, 3.0, 4.0, 5.0]]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    model_name: str
    predictions: List[float]
    shape: List[int]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    models_loaded: List[str]
    model_count: int


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Server status and list of loaded models
    """
    return HealthResponse(
        status="healthy",
        models_loaded=list(loaded_models.keys()),
        model_count=len(loaded_models)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions using a loaded model.
    
    Args:
        request: PredictionRequest with model_name and features
        
    Returns:
        PredictionResponse: Predictions and metadata
        
    Raises:
        HTTPException: If model not found or prediction fails
    """
    model_name = request.model_name
    
    # Validate model name (security check)
    model_filename = f"{model_name}.joblib"
    if not validate_model_filename(model_filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Allowed models: {[m.replace('.joblib', '') for m in ALLOWED_MODEL_NAMES]}"
        )
    
    # Check if model is loaded
    if model_name not in loaded_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {list(loaded_models.keys())}"
        )
    
    try:
        # Get model
        model = loaded_models[model_name]
        
        # Convert features to contiguous numpy array
        features = np.array(request.features, dtype=np.float64)
        features = np.ascontiguousarray(features)
        
        # Ensure 2D shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        logger.info(f"Prediction request: model={model_name}, shape={features.shape}")
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            
            # For binary classification, return class-1 probabilities
            if probabilities.shape[1] == 2:
                predictions = probabilities[:, 1].tolist()
            else:
                # For multiclass, return all probabilities
                predictions = probabilities.tolist()
        else:
            # Fallback to predict if predict_proba not available
            predictions = model.predict(features).tolist()
        
        return PredictionResponse(
            model_name=model_name,
            predictions=predictions,
            shape=list(features.shape)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ML Model Serving API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    port = int(os.getenv('PORT', '8000'))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
