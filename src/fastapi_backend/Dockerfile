# Dockerfile for the FastAPI Backend

FROM python:3.12-slim

# Set environment variables for non-interactive installation
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN pip install uv

# Copy the application source code, config, and assets into the container
COPY ./src/fastapi_backend ./src/fastapi_backend
COPY ./src/core ./src/core
COPY ./config.yaml ./config.yaml
COPY ./assets ./assets
COPY pyproject.toml ./

# Install dependencies using uv that are specific to the backend
RUN uv pip install --system --no-cache ".[backend]"

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Set the host to "0.0.0.0" to allow connections from outside the container.
CMD ["uvicorn", "src.fastapi_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
