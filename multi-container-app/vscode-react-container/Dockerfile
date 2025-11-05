FROM node:14

# Set the working directory
WORKDIR /app

# Install NVM
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
ENV NVM_DIR /root/.nvm
RUN ["/bin/bash", "-c", "source $NVM_DIR/nvm.sh && nvm install $(cat .nvmrc)"]

# Copy package.json and install dependencies
COPY package.json ./
RUN ["/bin/bash", "-c", "source $NVM_DIR/nvm.sh && npm install"]

# Copy the rest of the application code
COPY src ./src

# Expose the port the app runs on
EXPOSE 3000

# Command to run the server and client
CMD ["/bin/bash", "-c", "source $NVM_DIR/nvm.sh && node src/server.js & npm start --prefix ./src"]