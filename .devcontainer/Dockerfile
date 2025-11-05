# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Use bash for RUN instructions so that nvm can be sourced correctly
SHELL ["/bin/bash", "-c"]

# Install Node.js via nvm and ensure npm is upgraded for the React tooling
ARG NVM_VERSION=0.39.7
ARG NODE_VERSION=18.19.1
ARG NPM_VERSION=10.5.0
ENV NVM_DIR=/usr/local/share/nvm
ENV NODE_VERSION=${NODE_VERSION}

RUN mkdir -p "${NVM_DIR}" \
    && curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/v${NVM_VERSION}/install.sh" | bash \
    && source "${NVM_DIR}/nvm.sh" \
    && nvm install "${NODE_VERSION}" \
    && nvm alias default "${NODE_VERSION}" \
    && nvm use default \
    && npm install -g "npm@${NPM_VERSION}"

# Make Node.js available on PATH for all users and shells
ENV PATH="${NVM_DIR}/versions/node/v${NODE_VERSION}/bin:${PATH}"

# Install the QuickFIX Python package globally so it's available prior to venv creation
RUN pip install --no-cache-dir quickfix

# Set default workdir (VS Code devcontainer will override to workspace folder)
WORKDIR /workspace
