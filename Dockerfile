FROM nvidia/8.0-cudnn5-runtime-ubuntu16.04
MAINTAINER Martin Boos <mboos@hsr.ch>

RUN apt-get update -y && apt-get install -y git wget