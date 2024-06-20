#FROM  nvidia/cuda:10.1-cudnn7-devel-ubuntu20.04
FROM  nvidia/cuda:11.2.2-cudnn8-devel-ubuntu16.04

# update packages
RUN set -x && \
    apt-get update && \
    apt-get upgrade -y

# install command
RUN set -x && \
    apt-get install -y wget && \
    apt-get install -y sudo && \
    apt-get install -y vim && \
    apt-get install -y git && \
    apt-get install -y tmux 

# anaconda
RUN set -x
RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
RUN bash Anaconda3-2019.10-Linux-x86_64.sh -b
RUN rm Anaconda3-2019.10-Linux-x86_64.sh

# path setteing
ENV PATH $PATH:/root/anaconda3/bin

# RUN conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
#RUN apt install -y python3-pip
#RUN python3 -m pip install --upgrade pip
#RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html


#RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

WORKDIR /home/nakanose
#RUN conda update --all
ADD requirements.txt /home/nakanose
RUN pip install -r requirements.txt

# set character code
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
