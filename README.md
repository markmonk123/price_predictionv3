# Price Prediction v3

A comprehensive machine learning system for price prediction using imbalanced-learn ensemble models, with a full-stack architecture including Python model training, FastAPI serving, Node.js backend integration, and React frontend.

## 🏗️ Architecture

```
price_predictionv3/
├── src/                          # Python ML pipeline
│   ├── models/                   # Model training & utilities
│   │   ├── utils.py             # Helper functions
│   │   ├── ensemble_zoo.py      # Ensemble model training
│   │   └── __init__.py
│   └── api/                      # Model serving
│       └── fastapi_server.py    # FastAPI REST API
├── services/
│   ├── node-backend/            # Node.js connector service
│   │   ├── index.js             # Express server with Coinbase integration
│   │   ├── package.json
│   │   └── .env.example
│   └── react-frontend/          # React dashboard
│       ├── src/
│       │   ├── App.js           # Main React component
│       │   ├── App.css          # Styling
│       │   └── index.js
│       ├── public/
│       └── package.json
├── models/                       # Saved model artifacts (gitignored)
├── requirements.txt             # Python dependencies
└── README.md
```

## ✨ Features

### Machine Learning Pipeline
- **Imbalanced-Learn Integration**: Uses SMOTE, RandomUnderSampler, SMOTEENN for class balancing
- **Ensemble Zoo**: Three production-ready ensemble strategies:
  1. **Balanced Stacking**: SMOTE + Stacking Classifier
  2. **Balanced Voting**: RandomUnderSampler + Voting Classifier
  3. **Hybrid**: SMOTEENN + EasyEnsemble Classifier
- **Scaler Evaluation**: Automatically ranks top-5 scalers via cross-validation
- **Memory Efficient**: Contiguous numpy arrays, minimal copies
- **Deterministic**: Fixed random_state throughout for reproducibility
- **Parallel Processing**: Joblib-based parallelism for fast training

### Model Serving (FastAPI)
- **REST API**: `/health` and `/predict` endpoints
- **Input Validation**: Pydantic schemas with size limits
- **Security**: CORS configuration, request size limits, safe model loading
- **Production-Ready**: Environment-based configuration

### Node.js Backend
- **Coinbase Integration**: Connect via @coinbase/coinbase-sdk for market data
- **FIX Protocol Support**: Optional quickfix integration (skeleton provided)
- **Feature Normalization**: Prepares data for ML model
- **Proxy to Python**: Forwards predictions to FastAPI service

### React Frontend
- **Dashboard UI**: Clean, responsive interface
- **Real-Time Status**: Service health monitoring
- **Interactive Predictions**: Input market data and get predictions
- **Error Handling**: User-friendly error messages

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ (for ML training and serving)
- Node.js 16+ (for backend connector and frontend)
- npm or yarn

### Installation

#### 1. Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Node.js Backend

```bash
cd services/node-backend
npm install

# Configure environment
cp .env.example .env
# Edit .env and add your Coinbase API credentials
```

#### 3. React Frontend

```bash
cd services/react-frontend
npm install

# Configure environment (optional)
cp .env.example .env
```

## 📊 Training Models

### Prepare Your Data

Prepare a CSV file with features and a target column:

```csv
feature1,feature2,feature3,target
1.0,2.0,3.0,0
4.0,5.0,6.0,1
...
```

### Train Ensembles

```bash
# Activate Python environment
source venv/bin/activate

# Train models (will save to models/ directory)
python -m src.models.ensemble_zoo \
  --input data/training_data.csv \
  --target target \
  --output-dir models \
  --cv 5 \
  --random-state 42
```

This will:
1. Evaluate top-5 scalers via cross-validation
2. Build 3 ensemble models with imbalanced-learn
3. Train with stratified k-fold cross-validation
4. Save models to `models/` directory:
   - `ensemble_balanced_stacking.joblib`
   - `ensemble_voting.joblib`
   - `ensemble_hybrid.joblib`

### Model Training Options

```bash
python -m src.models.ensemble_zoo --help

Options:
  --input PATH           Path to input CSV file (required)
  --target COLUMN        Name of target column (default: 'target')
  --output-dir PATH      Directory for saved models (default: 'models')
  --cv N                 Cross-validation folds (default: 5)
  --random-state N       Random state for reproducibility
  --n-jobs N             Parallel jobs (default: -1, all cores)
```

## 🌐 Running Services

### 1. Start FastAPI Model Server

```bash
# Activate Python environment
source venv/bin/activate

# Start server
python -m src.api.fastapi_server

# Or with uvicorn directly
uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

**Environment Variables:**
- `MODEL_DIR`: Directory containing .joblib models (default: "models")
- `LOG_LEVEL`: Logging level (default: "INFO")
- `MAX_FEATURES`: Max features per request (default: 10000)
- `MAX_SAMPLES`: Max samples per request (default: 10000)
- `MAX_REQUEST_SIZE`: Max request body size in bytes (default: 10MB)
- `ALLOWED_ORIGINS`: CORS allowed origins (comma-separated)

**API Endpoints:**

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
    "model_name": "ensemble_balanced_stacking"
  }'
```

### 2. Start Node.js Backend

```bash
cd services/node-backend

# Make sure .env is configured
npm start
```

The backend will be available at `http://localhost:3000`

**Environment Variables (services/node-backend/.env):**
- `PORT`: Server port (default: 3000)
- `MODEL_SERVICE_URL`: Python FastAPI URL (default: http://localhost:8000)
- `COINBASE_API_KEY`: Your Coinbase API key
- `COINBASE_API_SECRET`: Your Coinbase API secret
- `ALLOWED_ORIGINS`: CORS origins for frontend

**API Endpoints:**

```bash
# Health check
curl http://localhost:3000/health

# Market data (placeholder)
curl http://localhost:3000/market/BTC-USD

# Prediction via Node backend
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "marketData": {
      "price": 45000,
      "volume": 1000000,
      "bid": 44950,
      "ask": 45050,
      "high": 46000,
      "low": 44000
    }
  }'
```

### 3. Start React Frontend

```bash
cd services/react-frontend
npm start
```

The frontend will open at `http://localhost:3001` (or 3000 if available)

## 🔒 Security Considerations

### Implemented Security Features

1. **No Secrets in Code**
   - All sensitive configuration via environment variables
   - `.env` files gitignored
   - `.env.example` templates provided

2. **Input Validation**
   - Pydantic schemas validate all API inputs
   - Size limits prevent memory exhaustion attacks
   - Type checking ensures numeric dtypes

3. **Safe Model Loading**
   - Models loaded only from specified `MODEL_DIR`
   - Path traversal prevention in model name validation
   - Joblib used instead of raw pickle (safer for sklearn objects)

4. **Request Limits**
   - Configurable request size limits
   - Max features and samples limits
   - Timeout configurations

5. **CORS Configuration**
   - Whitelist-based CORS allowed origins
   - Configurable per environment

6. **Deterministic Execution**
   - Fixed `random_state` throughout ML pipeline
   - Reproducible results for auditing

7. **Logging**
   - Structured logging with configurable levels
   - No sensitive data in logs
   - Error sanitization before client response

### Security Best Practices

**DO:**
- ✅ Store API keys in `.env` files (never commit these)
- ✅ Use HTTPS in production for all service communication
- ✅ Set restrictive CORS origins in production
- ✅ Monitor logs for suspicious activity
- ✅ Keep dependencies updated
- ✅ Use environment-specific configurations

**DON'T:**
- ❌ Commit `.env` files or API keys to git
- ❌ Expose internal error details to clients
- ❌ Use default credentials in production
- ❌ Allow unlimited request sizes
- ❌ Disable input validation

## 🧪 Testing

### Manual Testing

1. **Test Model Training**
```bash
# Create synthetic test data
python -c "
import pandas as pd
import numpy as np

np.random.seed(42)
X = np.random.randn(1000, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y
df.to_csv('test_data.csv', index=False)
"

# Train on synthetic data
python -m src.models.ensemble_zoo --input test_data.csv --cv 3
```

2. **Test FastAPI Server**
```bash
# Start server
python -m src.api.fastapi_server &

# Test health
curl http://localhost:8000/health

# Test prediction (after training)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]}'
```

3. **Test Node Backend**
```bash
cd services/node-backend
npm start &

curl http://localhost:3000/health
```

### Automated Testing (Recommended)

Create `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from src.api.fastapi_server import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_validation():
    response = client.post("/predict", json={
        "features": [[]]  # Empty features
    })
    assert response.status_code == 422  # Validation error
```

Run with pytest:
```bash
pytest tests/
```

## 📦 Dependencies

### Python (requirements.txt)

Conservative version ranges for maximum compatibility:

- `numpy>=1.21,<1.25`: Numerical computing
- `scipy>=1.7,<1.11`: Scientific computing
- `pandas>=1.3,<2.2`: Data manipulation
- `scikit-learn>=1.0,<1.3`: Machine learning
- `imbalanced-learn>=0.9,<0.13`: Imbalanced dataset handling
- `xgboost>=1.6,<2.0`: Gradient boosting (optional)
- `joblib>=1.1,<2.2`: Model serialization
- `fastapi>=0.90,<1.0`: Web framework
- `uvicorn>=0.18,<1.0`: ASGI server
- `pydantic>=1.8,<2.0`: Data validation
- `pytest>=7.0,<8.0`: Testing (optional)

### Node.js (services/node-backend/package.json)

- `@coinbase/coinbase-sdk`: Coinbase API integration
- `axios`: HTTP client
- `dotenv`: Environment variable management
- `express`: Web framework
- `cors`: CORS middleware
- `quickfix`: FIX protocol (optional)

### React (services/react-frontend/package.json)

- `react`: UI library
- `react-dom`: React DOM rendering
- `axios`: HTTP client
- `react-scripts`: Build tooling

## 🔧 Development

### Project Structure

```
src/
├── models/              # ML training pipeline
│   ├── utils.py        # Memory-efficient helpers, scaler ranking
│   ├── ensemble_zoo.py # 3 ensemble strategies with imbalanced-learn
│   └── __init__.py
└── api/
    └── fastapi_server.py  # REST API for model serving

services/
├── node-backend/       # Connector service
│   ├── index.js       # Express + Coinbase + FIX integration
│   └── package.json
└── react-frontend/     # Web dashboard
    ├── src/
    │   ├── App.js     # Main component
    │   └── App.css
    └── package.json
```

### Adding New Ensemble Models

1. Add model builder method to `EnsembleZoo` class in `src/models/ensemble_zoo.py`
2. Add model to `build_and_train_all()` method
3. Update FastAPI server model whitelist in `src/api/fastapi_server.py`
4. Retrain and redeploy

### Customizing Feature Engineering

Edit the `normalizeFeatures()` function in `services/node-backend/index.js` to match your model's expected features.

## 🐳 Docker Deployment (Optional)

The repository includes a `Dockerfile` for containerization:

```bash
# Build image
docker build -t price-prediction:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_DIR=/app/models \
  price-prediction:latest
```

## 📝 TODO / Future Enhancements

- [ ] Add pytest suite for automated testing
- [ ] Implement real Coinbase WebSocket streaming
- [ ] Add FIX protocol configuration examples
- [ ] Add model versioning and A/B testing
- [ ] Implement model monitoring and drift detection
- [ ] Add Prometheus metrics endpoint
- [ ] Create Docker Compose setup for full stack
- [ ] Add CI/CD pipeline configuration
- [ ] Implement authentication for production APIs

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📧 Support

For issues and questions, please open a GitHub issue.

---

**Built with:** Python, scikit-learn, imbalanced-learn, FastAPI, Node.js, React

**Security First:** No secrets in code, comprehensive input validation, production-ready defaults
