# Start from PyTorch base image with CUDA support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .[dev,docs]

# Copy the rest of the code
COPY . .

# Create directories for data, checkpoints, and logs
RUN mkdir -p data checkpoints logs

# Set up wandb directory
RUN mkdir -p /root/.wandb

# Default command (can be overridden)
CMD ["bash"] 