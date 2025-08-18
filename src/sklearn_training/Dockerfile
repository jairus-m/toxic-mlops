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

# Install dependencies using uv
RUN uv pip install --system --no-cache -e ./src/sklearn_training -e ./src/core

# Add the current directory to Python path so 'src' module can be found
ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "-c", "from src.sklearn_training.train_model import main; main()"]
