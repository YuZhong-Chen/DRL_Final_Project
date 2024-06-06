FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    freeglut3-dev \
    x11-apps \
    python-pip

RUN pip install gym==0.23.1 \
    pip install gym-super-mario-bros \
    pip install wandb
    
ENV SHELL /bin/bash
ENV PIP_ROOT_USER_ACTION=ignore
ENV DISPLAY=$DISPLAY
CMD ["/bin/bash"]