#!/bin/sh
# Build for linux/amd64 and push to the EPFL registry - run from the infra/ directory

set -eu

TARGET_PLATFORM="linux/amd64"
IMAGE_NAME="registry.rcp.epfl.ch/sacs-zu/faremark:latest"

docker buildx build \
    -f Dockerfile \
    --platform "$TARGET_PLATFORM" \
    -t "$IMAGE_NAME" \
    --load .

# run `docker login registry.rcp.epfl.ch` if not logged in
docker push "$IMAGE_NAME"

echo "Pushed $IMAGE_NAME"
