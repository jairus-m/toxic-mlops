# Dockerfile for the FastAPI Backend

FROM python:3.12-slim

# Set environment variables for non-interactive installation
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN pip install uv

COPY ./src/sklearn_training ./src/sklearn_training
COPY ./src/core ./src/core
COPY ./config.yaml ./config.yaml
COPY ./assets ./assets
COPY pyproject.toml ./

# Install dependencies using uv that are specific to the training
RUN uv pip install --system --no-cache ".[training]"

CMD ["python", "-c", "from src.sklearn_training.train_model import run_training; run_training()"]
