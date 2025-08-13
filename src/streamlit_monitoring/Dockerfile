# Dockerfile for the Streamlit Frontend

FROM python:3.12-slim

# Set environment variables for non-interactive installation
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN pip install uv

# Copy the entire src directory to preserve module structure
COPY ./src/streamlit_monitoring ./src/streamlit_monitoring
COPY ./src/core ./src/core
COPY ./config.yaml ./config.yaml
COPY ./assets ./assets
COPY pyproject.toml ./

RUN uv pip install --system --no-cache -e ".[monitoring]"

EXPOSE 8502

CMD ["streamlit", "run", "--server.port", "8502", "src/streamlit_monitoring/app.py"]
