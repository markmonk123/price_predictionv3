# price_predictionv3

This project is structured with a React frontend and a Python backend.

## Frontend

The `frontend` folder contains a minimal React application that retrieves the
current Bitcoin price using the `@coinbase/cdp-sdk`. The UI is rendered through
`public/index.html` with basic CSS styling.

## Backend

The `backend` folder hosts a simple Flask application. It demonstrates machine
learning with **numpy** and **scikit-learn**, exposing a `/predict` endpoint that
returns a sample prediction.
