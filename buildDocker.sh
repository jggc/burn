#!/bin/bash

#docker login registry.nationtech.io -u nationtech

IMAGE="registry.nationtech.io/inference-web"

docker build . -t ${IMAGE} -t $(basename ${IMAGE})
if [[ $? != 0 ]]; then
  exit
fi

echo -n "Do you want to push the built image to the nationtech repo? [y/N]: "
read answer
if [[ ${answer} == 'y' ]]; then
  docker push ${IMAGE}
else
  echo "No push"
fi

