# Price Prediction v3

A comprehensive machine learning system for cryptocurrency price prediction featuring imbalanced-aware ensemble models, FastAPI model serving, and Node.js/React integration for exchange connectivity.

## 🎯 Features

- **Imbalanced-Learn Ensemble Models**: Three sophisticated ensemble strategies for handling class imbalance
  - Balanced Stacking: SMOTE + Stacking (RF, GB, BalancedRF)
  - Balanced Voting: Multiple resampling strategies in voting ensemble
  - Hybrid Zoo: EasyEnsemble + GB pipeline
- **Automatic Scaler Evaluation**: Ranks scalers by cross-validation performance
- **FastAPI Model Serving**: Production-ready REST API with security features
- **Node.js Backend**: Integration with Coinbase SDK and FIX protocol connector
- **React Frontend**: Simple UI for testing predictions
- **Deterministic Training**: Reproducible results with fixed random seeds
- **Security Best Practices**: Input validation, request size limits, secure credential management

## 📁 Project Structure

```
price_predictionv3/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch_crypto_data.py    # Real historical data fetcher
│   │   └── generate_crypto_data.py # Realistic data generator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── utils.py              # Data preprocessing utilities
│   │   └── ensemble_zoo.py       # Ensemble model building
│   └── api/
│       ├── __init__.py
│       └── fastapi_server.py     # Model serving API
├── services/
│   ├── node-backend/
│   │   ├── package.json
│   │   └── index.js              # Exchange integration
│   └── react-frontend/
│       ├── package.json
│       └── src/
│           ├── App.js            # Frontend UI
│           └── App.css
├── models/                       # Trained models (gitignored)
│   ├── ensemble_balanced_stacking.joblib
│   ├── ensemble_voting.joblib
│   ├── ensemble_hybrid.joblib
│   └── ensemble_zoo_results.csv
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- pip and npm

### 1. Clone and Setup

```bash
git clone https://github.com/markmonk123/price_predictionv3.git
cd price_predictionv3
```

### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
# DO NOT commit .env file with real credentials!
```

### 4. Prepare Historical Data

**Option A: Generate Realistic Crypto Data (Recommended for Demo)**
```bash
# Generate 365 days of realistic Bitcoin-like price data
python -m src.data.generate_crypto_data --days 365 --price 45000 --output crypto_historical_data.csv

# This creates a CSV with:
# - 23 engineered features (OHLCV, moving averages, RSI, MACD, Bollinger Bands, etc.)
# - Binary target (next-day price increase prediction)
# - ~335 samples after feature engineering
```

**Option B: Fetch Real Historical Data (Requires Internet)**
```bash
# Fetch real data from Yahoo Finance
python -m src.data.fetch_crypto_data --symbol BTC-USD --days 365 --output crypto_historical_data.csv

# Note: Requires network access and yfinance package
```

### 5. Train Ensemble Models

```bash
# Train models using the historical crypto data
python -m src.models.ensemble_zoo --input crypto_historical_data.csv --target target --output-dir models

# This will create:
# - models/ensemble_balanced_stacking.joblib
# - models/ensemble_voting.joblib
# - models/ensemble_hybrid.joblib
# - models/ensemble_zoo_results.csv
```

### 6. Start FastAPI Server

```bash
# Start the model serving API
python -m uvicorn src.api.fastapi_server:app --host 0.0.0.0 --port 8000

# Or use the module directly
python src/api/fastapi_server.py

# API will be available at:
# - http://localhost:8000/docs (Swagger UI)
# - http://localhost:8000/health (Health check)
# - http://localhost:8000/predict (Prediction endpoint)
```

### 7. Start Node.js Backend

```bash
cd services/node-backend

# Install dependencies
npm install

# Start the server
npm start

# Backend will be available at http://localhost:3001
```

### 8. Start React Frontend

```bash
cd services/react-frontend

# Install dependencies
npm install

# Start development server
npm start

# Frontend will open at http://localhost:3000
```

## 📊 Model Training Details

### Historical Data Features

The system generates/fetches cryptocurrency historical data with comprehensive feature engineering:

**Price Features (OHLCV)**
- Open, High, Low, Close prices
- Trading volume

**Return Features**
- 1-day, 7-day, and 30-day returns

**Moving Averages**
- 7-day and 30-day moving averages
- MA ratio (7-day / 30-day)

**Volatility Indicators**
- 7-day and 30-day rolling volatility

**Technical Indicators**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands (upper, lower, width)

**Price Patterns**
- High/Low ratio
- Close/Open ratio
- Volume ratios

**Target Variable**
- Binary classification: 1 = next-day price increase, 0 = decrease/flat
- Creates naturally imbalanced dataset suitable for ensemble strategies

### Ensemble Strategies

#### 1. Balanced Stacking Ensemble
- **Resampling**: SMOTE (Synthetic Minority Over-sampling)
- **Base Estimators**:
  - Random Forest (50 trees, max_depth=10)
  - Gradient Boosting (50 estimators, max_depth=5)
  - Balanced Random Forest (50 trees, max_depth=10)
- **Final Estimator**: Logistic Regression
- **Scaler**: Auto-selected from top-5 candidates

#### 2. Balanced Voting Ensemble
- **Pipelines**:
  1. SMOTE + Random Forest
  2. Random Under-Sampling + Gradient Boosting
  3. SMOTEENN + Balanced Random Forest
- **Voting**: Soft voting (probability-based)
- **Scaler**: Auto-selected from top-5 candidates

#### 3. Hybrid Zoo Ensemble
- **Pipelines**:
  1. EasyEnsemble (10 estimators)
  2. SMOTE + Gradient Boosting
- **Voting**: Soft voting
- **Scaler**: Auto-selected from top-5 candidates

### Scaler Evaluation

The system automatically evaluates four scalers:
- StandardScaler (assumes Gaussian distribution)
- RobustScaler (robust to outliers using median/IQR)
- MinMaxScaler (scales to [0, 1] range)
- MaxAbsScaler (scales by maximum absolute value)

Top-5 scalers are tested with each ensemble via cross-validation, and the best performer is selected.

### Training Parameters

```python
build_ensembles(
    X, y,
    output_dir='models',      # Model save directory
    cv_splits=5,              # Cross-validation folds
    scoring='roc_auc',        # Evaluation metric
    n_jobs=1,                 # Parallelism (avoid nested)
    random_state=42           # Deterministic seed
)
```

## 🔒 Security Best Practices

### Credential Management
- ✅ **DO**: Store credentials in `.env` file
- ✅ **DO**: Use `.env.example` as a template
- ❌ **DON'T**: Commit `.env` file to version control
- ❌ **DON'T**: Hardcode API keys in source code

### API Security
- **Request Size Limiting**: Configurable via `MAX_REQUEST_SIZE` (default: 1MB)
- **CORS Configuration**: Restrict origins via `ALLOWED_ORIGINS`
- **Filename Validation**: Whitelist model names to prevent directory traversal
- **Input Validation**: Pydantic models enforce type checking and validation
- **Error Handling**: Detailed logging without exposing sensitive information

### Production Recommendations
1. **Use HTTPS/TLS** for all communications
2. **Implement rate limiting** on API endpoints
3. **Use environment-specific configurations** (dev/staging/prod)
4. **Enable authentication/authorization** for sensitive endpoints
5. **Regular security audits** of dependencies
6. **Monitor and log** all API requests
7. **Use secrets management** (e.g., AWS Secrets Manager, HashiCorp Vault)

## 🧪 Testing

### Test FastAPI Server

```bash
# Check health
curl http://localhost:8000/health

# Make prediction (requires trained models)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "ensemble_balanced_stacking",
    "features": [[45123.45, 123.456, 45120.0, 45125.0, 45500.0, 44800.0, 45000.0]]
  }'
```

### Test Node.js Backend

```bash
# Check health
curl http://localhost:3001/health

# Test prediction pipeline
curl http://localhost:3001/test-prediction

# Forward tick data
curl -X POST http://localhost:3001/forward-tick \
  -H "Content-Type: application/json" \
  -d '{
    "price": 45123.45,
    "volume": 123.456,
    "bid": 45120.0,
    "ask": 45125.0,
    "high": 45500.0,
    "low": 44800.0,
    "open": 45000.0
  }'
```

### Historical Data Generation

```bash
# Generate realistic cryptocurrency data with proper OHLCV patterns
python -m src.data.generate_crypto_data --days 365 --price 45000 --output crypto_historical_data.csv

# The generator creates:
# - Realistic price movements using geometric Brownian motion
# - Proper OHLC relationships
# - Log-normal volume distribution
# - 23 technical features
# - Binary classification target

# Inspect the generated data
python << 'EOF'
import pandas as pd
df = pd.read_csv('crypto_historical_data.csv', index_col=0, parse_dates=True)
print(f"Shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")
EOF

# Train models on this data
python -m src.models.ensemble_zoo --input crypto_historical_data.csv --target target
```

### Alternative: Legacy Synthetic Data

```python
# For comparison, you can also create simple synthetic imbalanced data
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Generate balanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=7,
    n_redundant=2,
    n_classes=2,
    weights=[0.7, 0.3],  # Imbalanced
    random_state=42
)

# Save to CSV
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y
df.to_csv('synthetic_data.csv', index=False)

# Train models
# python -m src.models.ensemble_zoo --input synthetic_data.csv --target target
```

## 🛠️ Development

### Python Development

```bash
# Install in development mode
pip install -e .

# Run linting (if configured)
flake8 src/

# Type checking (if configured)
mypy src/
```

### Node.js Development

```bash
# Use nodemon for auto-reload
cd services/node-backend
npm run dev
```

### React Development

```bash
# Start with hot reload
cd services/react-frontend
npm start
```

## 📦 Model Persistence

Models are saved with **joblib compression level 3** for optimal storage efficiency:

```python
joblib.dump(model, 'model.joblib', compress=3)
model = joblib.load('model.joblib')
```

### Model Files
- `ensemble_balanced_stacking.joblib` (~10-50 MB)
- `ensemble_voting.joblib` (~20-80 MB)
- `ensemble_hybrid.joblib` (~15-60 MB)
- `ensemble_zoo_results.csv` (metadata and scores)

## 🔍 Monitoring and Logging

All components include structured logging:

```python
# Python logging
logger.info("Model loaded successfully")
logger.error("Prediction failed", exc_info=True)

# Node.js console logging
console.log('✓ Operation successful');
console.error('✗ Operation failed:', error);
```

### Log Levels
- **INFO**: Normal operations, model loading, predictions
- **WARNING**: Recoverable errors, missing optional components
- **ERROR**: Critical errors requiring attention
- **DEBUG**: Detailed diagnostic information

## 🌐 API Documentation

### FastAPI Endpoints

#### GET /health
Returns server health and loaded models.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": ["ensemble_balanced_stacking", "ensemble_voting", "ensemble_hybrid"],
  "model_count": 3
}
```

#### POST /predict
Make predictions using a loaded model.

**Request:**
```json
{
  "model_name": "ensemble_balanced_stacking",
  "features": [[45123.45, 123.456, 45120.0, 45125.0, 45500.0, 44800.0, 45000.0]]
}
```

**Response:**
```json
{
  "model_name": "ensemble_balanced_stacking",
  "predictions": [0.7234],
  "shape": [1, 7]
}
```

### Node.js Endpoints

#### GET /health
Backend health check.

#### POST /forward-tick
Forward tick data for prediction.

**Request:**
```json
{
  "price": 45123.45,
  "volume": 123.456,
  "bid": 45120.0,
  "ask": 45125.0,
  "high": 45500.0,
  "low": 44800.0,
  "open": 45000.0,
  "model_name": "ensemble_balanced_stacking"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **imbalanced-learn**: SMOTE and ensemble methods for imbalanced datasets
- **scikit-learn**: Core machine learning algorithms
- **FastAPI**: Modern, fast web framework for APIs
- **React**: UI library for building user interfaces
- **Express**: Fast, unopinionated web framework for Node.js

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Always test thoroughly before using in production environments. Cryptocurrency trading involves significant risk. Past performance does not guarantee future results.
