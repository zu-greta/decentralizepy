#!/bin/sh
# Build for linux/amd64 and push to EPFL registry

TARGET_PLATFORM="linux/amd64"
IMAGE_NAME="registry.rcp.epfl.ch/sacs-zu/faremark-2:latest"

docker buildx build \
    -f Dockerfile \
    --platform $TARGET_PLATFORM \
    -t $IMAGE_NAME \
    --load .

# docker login registry.rcp.epfl.ch
# docker push registry.rcp.epfl.ch/sacs-zu/faremark-2:latest