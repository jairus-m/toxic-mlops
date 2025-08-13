# Dockerfile for the Streamlit Frontend

FROM python:3.12-slim

# Set environment variables for non-interactive installation
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN pip install uv

COPY ./src/streamlit_frontend ./src/streamlit_frontend
COPY ./src/core ./src/core
COPY ./config.yaml ./config.yaml
COPY ./assets ./assets
COPY pyproject.toml ./

RUN uv pip install --system --no-cache -e ".[frontend]"

EXPOSE 8501

CMD ["streamlit", "run", "--server.port", "8501", "src/streamlit_frontend/app.py"]
