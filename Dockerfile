# Use a minimal Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VERSION=1.8.2

# Install system dependencies
RUN apt-get update && apt-get install -y curl build-essential

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Create and set working directory
WORKDIR /app

# Copy only what’s needed for dependency resolution first (for caching)
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev for lighter image)
RUN poetry install --no-root --only main

# Copy the rest of the source code
COPY . .

# Expose port
EXPOSE 8000

# Default run command
CMD ["poetry", "run", "llmp-api"]
