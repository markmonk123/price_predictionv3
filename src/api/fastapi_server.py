"""
FastAPI Model Serving Endpoint.

Provides secure model serving with:
- /health endpoint for service health checks
- /predict endpoint with Pydantic validation
- CORS configuration via environment variables
- Request size limits and input validation
- Safe model loading with joblib
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants from environment with secure defaults
MODELS_DIR = Path(os.getenv('MODELS_DIR', 'models'))
MAX_FEATURES = int(os.getenv('MAX_FEATURES', '1000'))
MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '100'))
MAX_REQUEST_SIZE = int(os.getenv('MAX_REQUEST_SIZE', '10485760'))  # 10MB default
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(',')

# Initialize FastAPI app
app = FastAPI(
    title="Price Prediction Model Server",
    description="Secure model serving endpoint for imbalanced-learn ensemble models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global model storage
LOADED_MODELS: Dict[str, Any] = {}


class PredictionInput(BaseModel):
    """
    Input schema for prediction requests.
    
    Accepts either a list of feature values (single prediction)
    or a list of lists (batch prediction).
    """
    features: Union[List[float], List[List[float]]] = Field(
        ...,
        description="Feature values for prediction (single or batch)"
    )
    model_name: Optional[str] = Field(
        default="ensemble_hybrid",
        description="Model name to use for prediction"
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate feature dimensions and types."""
        if not v:
            raise ValueError("Features cannot be empty")
        
        # Check if batch or single prediction
        if isinstance(v[0], list):
            # Batch prediction
            if len(v) > MAX_BATCH_SIZE:
                raise ValueError(
                    f"Batch size {len(v)} exceeds maximum {MAX_BATCH_SIZE}"
                )
            
            # Check feature count for each sample
            feature_counts = [len(sample) for sample in v]
            if not all(fc == feature_counts[0] for fc in feature_counts):
                raise ValueError("All samples must have same number of features")
            
            if feature_counts[0] > MAX_FEATURES:
                raise ValueError(
                    f"Feature count {feature_counts[0]} exceeds maximum {MAX_FEATURES}"
                )
            
            # Validate all values are numeric
            for sample in v:
                if not all(isinstance(x, (int, float)) for x in sample):
                    raise ValueError("All feature values must be numeric")
        else:
            # Single prediction
            if len(v) > MAX_FEATURES:
                raise ValueError(
                    f"Feature count {len(v)} exceeds maximum {MAX_FEATURES}"
                )
            
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("All feature values must be numeric")
        
        return v
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name to prevent path traversal."""
        if v and ('/' in v or '\\' in v or '..' in v):
            raise ValueError("Invalid model name")
        return v


class PredictionOutput(BaseModel):
    """Output schema for prediction responses."""
    predictions: List[int] = Field(
        ...,
        description="Predicted class labels"
    )
    probabilities: Optional[List[List[float]]] = Field(
        None,
        description="Prediction probabilities for each class"
    )
    model_name: str = Field(
        ...,
        description="Model used for prediction"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    models_loaded: List[str]
    message: str


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Middleware to limit request body size."""
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": f"Request too large. Maximum size: {MAX_REQUEST_SIZE} bytes"}
        )
    return await call_next(request)


def safe_load_model(model_path: Path) -> Any:
    """
    Safely load a joblib model with validation.
    
    Args:
        model_path: Path to model file
    
    Returns:
        Loaded model object
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model loading fails
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not model_path.is_file():
        raise ValueError(f"Model path is not a file: {model_path}")
    
    # Validate file extension
    if model_path.suffix != '.joblib':
        raise ValueError(f"Invalid model file extension: {model_path.suffix}")
    
    try:
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully: {model_path.name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        raise ValueError(f"Model loading failed: {e}")


def load_available_models() -> Dict[str, Any]:
    """
    Load all available models from models directory.
    
    Returns:
        Dictionary mapping model names to loaded model objects
    """
    models = {}
    
    if not MODELS_DIR.exists():
        logger.warning(f"Models directory not found: {MODELS_DIR}")
        return models
    
    model_files = [
        'ensemble_balanced_stacking.joblib',
        'ensemble_voting.joblib',
        'ensemble_hybrid.joblib',
    ]
    
    for model_file in model_files:
        model_path = MODELS_DIR / model_file
        model_name = model_path.stem  # Remove .joblib extension
        
        try:
            models[model_name] = safe_load_model(model_path)
        except Exception as e:
            logger.warning(f"Could not load {model_file}: {e}")
    
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    return models


@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    global LOADED_MODELS
    logger.info("Starting up Model Server...")
    logger.info(f"Models directory: {MODELS_DIR.absolute()}")
    logger.info(f"CORS origins: {CORS_ORIGINS}")
    logger.info(f"Max features: {MAX_FEATURES}")
    logger.info(f"Max batch size: {MAX_BATCH_SIZE}")
    
    LOADED_MODELS = load_available_models()
    
    if not LOADED_MODELS:
        logger.warning("No models loaded! Predictions will fail.")
    else:
        logger.info(f"Successfully loaded models: {list(LOADED_MODELS.keys())}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and list of loaded models.
    """
    is_healthy = len(LOADED_MODELS) > 0
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        models_loaded=list(LOADED_MODELS.keys()),
        message=f"{len(LOADED_MODELS)} model(s) loaded and ready" if is_healthy else "No models loaded"
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Prediction endpoint with input validation.
    
    Accepts feature vectors and returns predictions with optional probabilities.
    
    Args:
        input_data: Validated prediction input
    
    Returns:
        Predictions and probabilities
    
    Raises:
        HTTPException: If model not found or prediction fails
    """
    # Check if any models are loaded
    if not LOADED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models available for prediction"
        )
    
    # Get requested model or default
    model_name = input_data.model_name
    if model_name not in LOADED_MODELS:
        # Fall back to first available model
        model_name = list(LOADED_MODELS.keys())[0]
        logger.warning(
            f"Requested model '{input_data.model_name}' not found. "
            f"Using '{model_name}' instead."
        )
    
    model = LOADED_MODELS[model_name]
    
    # Prepare input array
    try:
        features = input_data.features
        
        # Detect single vs batch prediction
        if isinstance(features[0], list):
            # Batch prediction
            X = np.array(features, dtype=np.float64)
        else:
            # Single prediction - reshape to 2D
            X = np.array([features], dtype=np.float64)
        
        logger.debug(f"Input shape: {X.shape}")
        
        # Make predictions
        predictions = model.predict(X)
        predictions_list = predictions.tolist()
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X)
                probabilities = proba.tolist()
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
        
        logger.info(
            f"Prediction successful: model={model_name}, "
            f"samples={X.shape[0]}, features={X.shape[1]}"
        )
        
        return PredictionOutput(
            predictions=predictions_list,
            probabilities=probabilities,
            model_name=model_name
        )
    
    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/models")
async def list_models():
    """
    List available models.
    
    Returns:
        Dictionary of available model names and their status
    """
    return {
        "available_models": list(LOADED_MODELS.keys()),
        "count": len(LOADED_MODELS)
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv('LOG_LEVEL', 'info').lower()
    )
