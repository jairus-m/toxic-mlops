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

RUN uv pip install --system --no-cache -e ./src/streamlit_monitoring -e ./src/core

# Add the current directory to Python path so 'src' module can be found
ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 8502

CMD ["streamlit", "run", "--server.port", "8502", "src/streamlit_monitoring/app.py"]
