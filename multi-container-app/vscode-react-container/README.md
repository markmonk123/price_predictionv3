# README for VS Code React Container

This directory contains the setup for the React application that runs in a Docker container. Below are the instructions and details for building and running the application.

## Getting Started

To get started with the React application, follow these steps:

1. **Clone the Repository**: Ensure you have the entire multi-container application repository cloned to your local machine.

2. **Navigate to the Directory**: Change into the `vscode-react-container` directory.

   ```bash
   cd vscode-react-container
   ```

3. **Build the Docker Image**: Use the following command to build the Docker image for the React application.

   ```bash
   docker build -t vscode-react-app .
   ```

4. **Run the Docker Container**: After building the image, you can run the container using:

   ```bash
   docker run -p 3000:3000 vscode-react-app
   ```

   This will start the React application and expose it on port 3000.

## Application Structure

- **Dockerfile**: Contains the instructions to build the React application environment.
- **package.json**: Lists the Node.js dependencies and scripts for the React application.
- **nvmrc**: Specifies the Node.js version to be used in the container.
- **src/**: Contains the source code for the application.
  - **server.js**: Entry point for the server-side application.
  - **client.js**: Entry point for the client-side application.
  - **App.jsx**: Main React component that serves as the root of the application.

## Running the Application

Once the container is running, you can access the application by navigating to `http://localhost:3000` in your web browser.

## Additional Notes

- Ensure that Docker is installed and running on your machine.
- You may need to adjust firewall settings to allow access to the specified port.

For further assistance, please refer to the main project README or the documentation for Docker and React.