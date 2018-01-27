FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER Martin Boos <mboos@hsr.ch>

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get update -y &&\
    apt-get install -y git wget software-properties-common vim nano

#RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash &&\
#    apt-get install -y git-lfs
#
#RUN git lfs install &&\
#    git lfs track "*.h5" &&\
#    git add .gitattributes
#    add-apt-repository ppa:jonathonf/python-3.6 &&\
#    apt-get update -y &&\
#    apt-get install -y python3.6 python3-pip

#ENTRYPOINT chmod +x ./execute.sh && ./execute.sh
CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"
#ENTRYPOINT ["/bin/bash"]
