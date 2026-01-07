# Price Prediction v3 - Imbalanced-Learn Ensemble Zoo

ML-powered price prediction system with imbalanced-learn ensembles, FastAPI model serving, and Node.js/React frontend connectors.

## 🎯 Features

- **Ensemble Zoo**: Three production-ready ensemble models
  - Balanced Stacking Ensemble (SMOTE + Stacking)
  - Balanced Voting Ensemble (SMOTEENN + Voting)
  - Hybrid Ensemble (EasyEnsemble + Gradient Boosting)
- **Automated Scaler Evaluation**: Top-5 scaler ranking with cross-validation
- **FastAPI Model Server**: Secure REST API for predictions with Pydantic validation
- **Node.js Backend**: Coinbase SDK integration and FIX protocol support
- **React Frontend**: Minimal UI for predictions and service monitoring
- **Conservative Dependencies**: Version constraints for maximum compatibility
- **Security-First Design**: No secrets in code, input validation, request limits

## 📁 Project Structure

```
price_predictionv3/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── utils.py              # numpy helpers, scaler ranking
│   │   └── ensemble_zoo.py       # 3 ensembles, training CLI
│   └── api/
│       └── fastapi_server.py     # Model serving endpoint
├── services/
│   ├── node-backend/
│   │   ├── package.json
│   │   └── index.js              # Coinbase connector, FIX skeleton
│   └── react-frontend/
│       ├── package.json
│       ├── public/
│       │   └── index.html
│       └── src/
│           ├── index.js
│           ├── index.css
│           ├── App.js            # Main UI component
│           └── App.css
├── models/                       # Saved models (created during training)
│   └── ensemble_zoo_results.csv  # Generated on first run
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- pip and npm

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models (First Run)

Train the ensemble zoo on your data with a `target` column:

```bash
python -m src.models.ensemble_zoo data.csv --target-col target --cv 5
```

**Example with synthetic data:**

```python
# Create synthetic dataset for testing
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.7, 0.3],  # Imbalanced
    random_state=42
)

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
df['target'] = y
df.to_csv('data.csv', index=False)
```

Then train:

```bash
python -m src.models.ensemble_zoo data.csv
```

**Output:**
- Trained models saved to `models/ensemble_*.joblib`
- Performance report in `models/ensemble_zoo_results.csv`

### 3. Start FastAPI Model Server

```bash
# Set environment variables (optional)
export MODELS_DIR=models
export API_PORT=8000
export LOG_LEVEL=INFO
export CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Start server
python -m src.api.fastapi_server

# Or with uvicorn directly
uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
- `GET /health` - Service health check
- `POST /predict` - Single or batch predictions
- `GET /models` - List available models

**Example prediction request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.2, 3.4, 5.6, 7.8, 9.0, 1.1, 2.2, 3.3, 4.4, 5.5, 
                 6.6, 7.7, 8.8, 9.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "model_name": "ensemble_hybrid"
  }'
```

### 4. Start Node.js Backend

```bash
cd services/node-backend

# Install dependencies
npm install

# Create .env file (DO NOT commit this)
cat > .env << EOF
NODE_PORT=3001
COINBASE_API_KEY=your_api_key_here
COINBASE_API_SECRET=your_api_secret_here
ML_SERVICE_URL=http://localhost:8000
CORS_ORIGIN=http://localhost:3000
USE_TLS=false
EOF

# Start server
npm start

# Or with nodemon for development
npm run dev
```

**Endpoints:**
- `GET /health` - Backend health check
- `GET /market/:symbol` - Get market data (stub)
- `POST /predict` - Normalized prediction
- `POST /predict/batch` - Batch predictions
- `GET /ml-service/health` - Proxy to ML service

### 5. Start React Frontend

```bash
cd services/react-frontend

# Install dependencies
npm install

# Create .env file (optional)
echo "REACT_APP_BACKEND_URL=http://localhost:3001" > .env

# Start development server
npm start
```

Access the UI at `http://localhost:3000`

## 🔒 Security Considerations

### Environment Variables (Required)

Create `.env` files for each service - **NEVER commit these!**

**Python (.env in root):**
```bash
MODELS_DIR=models
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
MAX_FEATURES=1000
MAX_BATCH_SIZE=100
MAX_REQUEST_SIZE=10485760
```

**Node.js (.env in services/node-backend):**
```bash
NODE_PORT=3001
COINBASE_API_KEY=<your_key>
COINBASE_API_SECRET=<your_secret>
ML_SERVICE_URL=http://localhost:8000
CORS_ORIGIN=http://localhost:3000
USE_TLS=true  # Enable for production
```

**React (.env in services/react-frontend):**
```bash
REACT_APP_BACKEND_URL=http://localhost:3001
```

### Security Features

1. **No Secrets in Code**: All API keys via environment variables
2. **Pydantic Validation**: Input schema validation with size limits
3. **Request Size Limits**: Prevent DoS attacks (default 10MB)
4. **CORS Configuration**: Restrict allowed origins
5. **Safe Model Loading**: Path validation, joblib security
6. **Deterministic Training**: Fixed `random_state` for reproducibility
7. **Memory Efficiency**: Contiguous numpy arrays, minimal copies
8. **TLS Support**: HTTPS for production Node backend

### Production Deployment Notes

- Use HTTPS/TLS for all services
- Set strict CORS origins (not `*`)
- Use environment-specific `.env` files
- Enable request rate limiting
- Monitor logs for suspicious activity
- Regularly update dependencies for security patches
- Use Docker secrets or vault systems for sensitive data
- Run services with minimal privileges (non-root users)

## 📊 Model Training Details

### Ensemble Strategies

1. **Balanced Stacking**
   - Base: Balanced Random Forest, Gradient Boosting, Decision Tree
   - Sampling: SMOTE (oversampling)
   - Meta-learner: Logistic Regression

2. **Balanced Voting**
   - Estimators: Balanced Random Forest, Gradient Boosting, Logistic Regression
   - Sampling: SMOTEENN (combined over/under sampling)
   - Voting: Soft voting

3. **Hybrid (EasyEnsemble + GB)**
   - EasyEnsembleClassifier with Gradient Boosting base
   - Handles imbalance through bagging + undersampling

### Scaler Evaluation

Top-5 scalers evaluated with cross-validation:
- StandardScaler
- MinMaxScaler
- RobustScaler
- MaxAbsScaler
- QuantileTransformer

Best scaler automatically selected based on F1-weighted score.

### Training Parameters

```bash
# CLI arguments
--target-col     # Target column name (default: 'target')
--cv             # CV folds (default: 5)
--scoring        # Metric (default: 'f1_weighted')
--random-state   # Random seed (default: 42)
--n-jobs         # Parallel jobs (default: -1, all CPUs)
```

## 🧪 Testing

### Test FastAPI Server

```bash
# Check health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]]}'
```

### Test Node Backend

```bash
# Check health
curl http://localhost:3001/health

# Get market data (stub)
curl http://localhost:3001/market/BTC-USD

# Test prediction
curl -X POST http://localhost:3001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "marketData": {
      "price": 50000,
      "volume": 1000000,
      "bid": 49900,
      "ask": 50100,
      "high": 51000,
      "low": 49000
    }
  }'
```

## 📦 Dependencies

### Python

See `requirements.txt` for conservative version ranges:
- numpy, scipy, pandas
- scikit-learn, imbalanced-learn
- xgboost (optional)
- fastapi, uvicorn, pydantic
- joblib, python-dotenv

### Node.js

See `services/node-backend/package.json`:
- express, axios, body-parser
- @coinbase/coinbase-sdk
- dotenv
- quickfix (optional, for FIX protocol)

### React

See `services/react-frontend/package.json`:
- react, react-dom
- axios
- react-scripts

## 🐳 Docker Support

Existing `Dockerfile` can be updated to support the new structure:

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install Node backend dependencies
WORKDIR /app/services/node-backend
RUN npm install --production

WORKDIR /app

# Expose ports
EXPOSE 8000 3001

# Default command (can override in docker-compose)
CMD ["python", "-m", "src.api.fastapi_server"]
```

## 🔧 Troubleshooting

### Models not loading

1. Ensure you've trained models first: `python -m src.models.ensemble_zoo data.csv`
2. Check `models/` directory exists and contains `.joblib` files
3. Verify `MODELS_DIR` environment variable points to correct location

### CORS errors in browser

1. Check FastAPI `CORS_ORIGINS` includes your frontend URL
2. Ensure Node backend `CORS_ORIGIN` is set correctly
3. Verify all services are running on expected ports

### Coinbase API not working

1. Set `COINBASE_API_KEY` and `COINBASE_API_SECRET` in `.env`
2. Implement actual SDK integration in `services/node-backend/index.js`
3. See Coinbase SDK docs: https://docs.cdp.coinbase.com/

### Port conflicts

Change ports in `.env` files:
- Python: `API_PORT=8001`
- Node: `NODE_PORT=3002`
- React: `PORT=3000` (React default)

## 📝 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

This is a demonstration/educational project. Not financial advice. Do not use in production without thorough testing and security audits.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📧 Support

For issues and questions, please open a GitHub issue in the repository.

---

**Built with:** imbalanced-learn • scikit-learn • FastAPI • React • Node.js • Express
