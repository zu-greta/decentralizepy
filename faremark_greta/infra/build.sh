#!/bin/sh
# Build for linux/amd64 and push to the EPFL registry.
# Run from the infra/ directory (it expects Dockerfile + requirements.txt here).

set -eu

TARGET_PLATFORM="linux/amd64"
# Single source of truth for the image name. submit_experiment.sh reads the
# same value, so build and submit can never drift apart again.
IMAGE_NAME="registry.rcp.epfl.ch/sacs-zu/faremark:latest"

docker buildx build \
    -f Dockerfile \
    --platform "$TARGET_PLATFORM" \
    -t "$IMAGE_NAME" \
    --load .

# Run `docker login registry.rcp.epfl.ch` once if you are not logged in.
docker push "$IMAGE_NAME"

echo "Pushed $IMAGE_NAME"
