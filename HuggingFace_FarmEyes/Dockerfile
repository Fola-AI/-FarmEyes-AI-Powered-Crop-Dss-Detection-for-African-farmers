# =============================================================================
# FarmEyes - HuggingFace Spaces Dockerfile (Transformers Version)
# =============================================================================

FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=7860
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Fix numpy version first (must be <2.0 for ultralytics compatibility)
RUN pip install "numpy<2.0"

# Install ultralytics
RUN pip install ultralytics

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Ensure numpy stays compatible
RUN pip install "numpy<2.0"

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/uploads /app/temp

EXPOSE 7860

CMD ["python", "main.py"]
