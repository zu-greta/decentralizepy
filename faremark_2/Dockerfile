# Dockerfile
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the FareMark source code (optional, can also mount from NFS)
# We'll assume code is mounted or cloned at runtime, so this is optional.
# If you want to bake code into the image, uncomment:
# COPY . /workspace/faremark
# WORKDIR /workspace/faremark

# Default command (overridden in submit)
CMD ["/bin/bash"]