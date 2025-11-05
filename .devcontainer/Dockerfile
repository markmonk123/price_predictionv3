# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Install system dependencies if needed (none currently)

# Install the QuickFIX Python package globally
RUN pip install --no-cache-dir quickfix

# Set default workdir (VS Code devcontainer will override to workspace folder)
WORKDIR /workspace
