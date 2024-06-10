FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ARG USERNAME=user
ARG USERID=1000
ARG USERGID=${USERID}
ARG SHELL=bash

LABEL org.opencontainers.image.authors="assume"
LABEL shell="${SHELL}"

ENV USERNAME=${USERNAME}
ENV SHELL=/bin/${SHELL}
ENV TERM=xterm-256color
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV PYTHONUNBUFFERED=1

ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && \
    apt-get install -y \
    sudo \
    vim \
    curl \
    wget \
    tmux \
    htop \
    git \
    git-extras \
    gnupg2 \
    net-tools \
    locales \
    g++ \
    # OpenCV necessary libraries
    ffmpeg libsm6 libxext6 libglib2.0-dev libxrender-dev freeglut3-dev ffmpeg \
    # Car racing enrironment required libraries
    swig libgeos-dev

RUN echo "en_US.UTF-8 UTF-8" > /etc/local.gen && locale-gen en_US.UTF-8

RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends tzdata
RUN TZ=Asia/Taipei && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata 

RUN groupadd --gid ${USERGID} ${USERNAME} \
    && useradd --uid ${USERID} --gid ${USERGID} -m ${USERNAME} -s /bin/bash \
    # passwordless
    && apt-get update \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

USER ${USERNAME}
WORKDIR /home/${USERNAME}

RUN pip install ftfy regex tqdm matplotlib
RUN pip install git+https://github.com/openai/CLIP.git

RUN sudo apt autoremove -y && \
    sudo apt clean -y

SHELL ["/bin/bash", "-c"]
CMD ["bash"]