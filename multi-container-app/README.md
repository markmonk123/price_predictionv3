# Multi-Container Application

This project is a multi-container setup that includes a Python environment with Node.js and a Visual Studio Code web application using React. The application is structured into two main containers: one for the Python and Node.js environment and another for the React application.

## Project Structure

```
multi-container-app
├── python-node-container
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   ├── package.json
│   ├── nvmrc
│   └── README.md
├── vscode-react-container
│   ├── Dockerfile
│   ├── package.json
│   ├── nvmrc
│   ├── src
│   │   ├── server.js
│   │   ├── client.js
│   │   └── App.jsx
│   └── README.md
├── docker-compose.yml
└── README.md
```

## Getting Started

To get started with this multi-container application, follow the instructions below:

### Prerequisites

- Docker
- Docker Compose

### Setup

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Build the containers using Docker Compose:

   ```
   docker-compose up --build
   ```

### Running the Application

- The Python-Node container will run the backend application, which can be accessed at `http://localhost:5000`.
- The React application will be served from the React container, accessible at `http://localhost:3000`.

### Container Details

- **Python-Node Container**: This container is responsible for running the Python application and Node.js scripts. It includes all necessary dependencies defined in `requirements.txt` and `package.json`.

- **React Container**: This container serves the React application. It uses the scripts defined in its `package.json` to start the application.

### Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

### License

This project is licensed under the MIT License. See the LICENSE file for more details.