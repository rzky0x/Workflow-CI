FROM python:3.12.7-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY MLProject/conda.yaml /app/
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.6.0 \
    pandas==2.2.3 \
    numpy==2.1.3 \
    matplotlib==3.9.3 \
    seaborn==0.13.2

# Copy project files
COPY MLProject/ /app/

# Expose MLflow model serving port
EXPOSE 5002

# Default: serve the model (will be overridden by CI for training)
CMD ["mlflow", "models", "serve", "-m", "models:/wine-quality-model/1", "--host", "0.0.0.0", "--port", "5002", "--no-conda"]
