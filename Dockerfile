# Dockerfile to build a T2 image 
#
# git clone https://github.com/dguarino/T2.git
# cd T2
# docker build -t t2 .
# docker run -it -v `pwd`:/home/docker/T2 t2 /bin/bash
# source env/neurosci/bin/activate

FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update  # merge the following two lines once we've finished developing
RUN apt-get install -y git python python-numpy python-scipy python-matplotlib \
                                       ipython python-virtualenv

RUN useradd -ms /bin/bash docker
USER docker
ENV HOME=/home/docker
RUN mkdir $HOME/env; mkdir $HOME/packages

ENV VENV=$HOME/env/neurosci

RUN python -m virtualenv --system-site-packages $VENV

RUN $VENV/bin/pip install quantities==0.10.1 neo==0.4.0 ipython==2.4.1 parameters

WORKDIR /home/docker
