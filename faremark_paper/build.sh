#!/bin/sh

# Specify the target platform
TARGET_PLATFORM="linux/amd64"

docker buildx build -f Dockerfile --build-arg DUMMY='' \
    -t "registry.rcp.epfl.ch/sacs-zu/faremark:latest" \
    --platform $TARGET_PLATFORM \
    --secret id=my_env,src=.env \
    --load .

docker push registry.rcp.epfl.ch/sacs-zu/faremark:latest


