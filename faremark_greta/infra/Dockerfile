# Base image matches the lab cluster's CUDA toolchain. The pytorch/pytorch
# images already bundle a matching torch + torchvision
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# code cloned from git at run time (see submit_experiment.sh)
CMD ["/bin/bash"]
