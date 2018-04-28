FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER Martin Boos <mboos@hsr.ch>

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get update -y &&\
    apt-get install -y git wget software-properties-common vim nano python3-tk &&\
    apt update && apt install -y libsm6 libxext6 &&
    pip3 install pycocotools

#ENTRYPOINT chmod +x ./execute && ./execute
CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
