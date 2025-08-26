from flask import Flask, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Train a simple model on startup using numpy and scikit-learn
X = np.array([[0], [1], [2], [3], [4]])
y = np.array([0, 1, 2, 3, 4])
model = LinearRegression().fit(X, y)

@app.route('/predict')
def predict():
    """Return a basic prediction from the trained model."""
    next_val = model.predict(np.array([[5]])).item()
    return jsonify({"prediction": next_val})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
