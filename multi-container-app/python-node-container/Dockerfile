FROM python:3.9-slim

# Set environment variables for Node.js
ENV NVM_DIR=/root/.nvm
ENV NODE_VERSION=14.17.0

# Install dependencies
RUN apt-get update && apt-get install -y curl && \
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash && \
    . "$NVM_DIR/nvm.sh" && \
    nvm install $NODE_VERSION && \
    nvm use $NODE_VERSION && \
    npm install -g npm

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the Node.js package file and install Node.js dependencies
COPY package.json .
RUN npm install

# Copy the application code
COPY . .

# Expose the necessary ports
EXPOSE 5000 3000

# Command to run both the Python app and the Node.js app
CMD ["sh", "-c", "python app.py & npm start"]