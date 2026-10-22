# ============================================================
# Dockerfile for GCP Cloud Run - ML Emotion & Proctoring Service
# ============================================================
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies needed for OpenCV, audio processing, and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install CPU-only PyTorch first (reduces image size from ~4GB to ~1.2GB)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . .

# Ensure text emotion weights are present (auto-downloads in seconds on GCP if excluded from upload)
RUN if [ ! -s models/text_emotion_model/pytorch_model.bin ]; then \
        echo "Downloading text emotion model weights (~300MB) from Hugging Face..." && \
        mkdir -p models/text_emotion_model && \
        curl -L -o models/text_emotion_model/pytorch_model.bin \
        "https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/resolve/main/pytorch_model.bin"; \
    fi

# Cloud Run injects $PORT at runtime (defaults to 8080)
EXPOSE 8080

# Run FastAPI via Uvicorn listening on 0.0.0.0:$PORT
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
