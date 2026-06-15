FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /workspace/

ENV PYTHONPATH=/workspace
ENV RESULTS_DIR=/results