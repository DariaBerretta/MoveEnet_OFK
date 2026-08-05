FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04

ARG SOURCE_FOLDER=/usr/local/src
ARG BUILD_TYPE=Release
ARG YCM_VERSION=v0.15.1
ARG YARP_VERSION=v3.7.2
ARG EVENT_DRIVEN_VERSION=master
ARG HPE_VERSION=main

ENV DEBIAN_FRONTEND noninteractive
RUN apt update 

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     software-properties-common curl ca-certificates gnupg \
#  && add-apt-repository ppa:deadsnakes/ppa -y \
#  && apt-get update && apt-get install -y --no-install-recommends \
#     python3.12 python3.12-venv python3.12-dev \
#  && python3.12 -m ensurepip --upgrade \
#  && python3.12 -m pip install --upgrade pip setuptools wheel \
#  && python3.12 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv python3-pip \
    curl ca-certificates gnupg \
 && python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

RUN apt install -y build-essential libssl-dev software-properties-common \
    cmake cmake-curses-gui vim nano git sudo openssh-client git \
    libboost-all-dev libmysqlclient-dev ffmpeg libsm6 libxext6 libcanberra-gtk-module \
    apt-transport-https ca-certificates gnupg lsb-core swig \
    libgsl-dev libedit-dev libace-dev libeigen3-dev 
    #qtbase5-dev qt5-default qtdeclarative5-dev qtmultimedia5-dev qml-module-qtquick2 qml-module-qtquick-window2 qml-module-qtmultimedia qml-module-qtquick-dialogs qml-module-qtquick-controls
    
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN echo "deb [arch=amd64 trusted=yes] https://apt.prophesee.ai/dists/public/7l58osgr/ubuntu focal essentials" >> /etc/apt/sources.list;

RUN apt update
# RUN apt install -y python3.7 python3-pip python3.7-dev
# RUN pip3 install numpy
RUN apt install -y python3-pip
RUN apt install -y libopencv-dev python3-opencv
# RUN apt update && apt install -y metavision-*

RUN cd $SOURCE_FOLDER && \
    git clone --depth 1 --branch $YCM_VERSION https://github.com/robotology/ycm.git &&\
    cd ycm && mkdir build && cd build && \
    cmake .. && make install -j$(nproc)

RUN cd $SOURCE_FOLDER && \
git clone --depth 1  --branch $YARP_VERSION https://github.com/robotology/yarp.git &&\
cd yarp && mkdir build && cd build && \
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DYARP_COMPILE_BINDINGS=ON \
        -DCREATE_PYTHON=ON \
        .. && \
make install -j$(nproc)

EXPOSE 10000/tcp 10000/udp

ENV PYTHONPATH $SOURCE_FOLDER/yarp/build/lib/python3:$PYTHONPATH
ENV QT_X11_NO_MITSHM 1

RUN cd $SOURCE_FOLDER && \
    git clone --depth 1 --branch $EVENT_DRIVEN_VERSION https://github.com/robotology/event-driven.git && \
    cd event-driven && mkdir build && cd build && \
    cmake -DVLIB_ENABLE_TS=OFF .. && make install -j$(nproc)

RUN apt update

RUN cd $SOURCE_FOLDER && \
    git clone --depth 1 --branch $HPE_VERSION https://github.com/event-driven-robotics/hpe-core.git &&\
    cd hpe-core/core && mkdir build && cd build && \
    cmake .. && make install -j$(nproc)

# RUN python3 -m pip install -r $SOURCE_FOLDER/hpe-core/example/movenet/requirements.txt
RUN /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/python -m pip install --no-cache-dir \
        -r $SOURCE_FOLDER/hpe-core/example/movenet/requirements.txt 
ENV PYTHONPATH "${PYTHONPATH}:$SOURCE_FOLDER/hpe-core"
    
RUN cd $SOURCE_FOLDER && \
    git clone --branch main https://github.com/event-driven-robotics/EDPR-APRIL.git && \
    cd EDPR-APRIL && mkdir build && cd build && \
    cmake .. && make install -j$(nproc)

RUN apt install iproute2 -y

# Build and install OpenPose (CPU+GPU, no Python bindings needed)
RUN apt update && apt install -y \
    libprotobuf-dev \
    protobuf-compiler \
    libhdf5-dev \
    libatlas-base-dev \
    libgoogle-glog-dev \
    libgflags-dev
RUN cd $SOURCE_FOLDER && \
    git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git && \
    cd openpose && \
    git submodule update --init --recursive --remote && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_PYTHON=OFF \
          -DUSE_CUDNN=ON \
          -DDOWNLOAD_BODY_25_MODEL=OFF \
          -DDOWNLOAD_COCO_MODEL=OFF \
          -DDOWNLOAD_MPI_MODEL=OFF \
          -DDOWNLOAD_FACE_MODEL=OFF \
          -DDOWNLOAD_HAND_MODEL=OFF \
          .. && \
    make -j$(nproc) && \
    make install

ARG OPENPOSE_MODEL_REVISION=f4a22b0e6fa2a4a2b1e2d50bd589e8bb11ebea7c
ARG OPENPOSE_MODEL_SHA256=44e3d7ebd8c8b62d4366d67127f1b562611a9e8fd0f4f3cdeeb4bb4a6ed12be6

RUN MODEL_DIR=/usr/local/src/openpose/models/pose/body_25 && \
    MODEL_FILE="${MODEL_DIR}/pose_iter_584000.caffemodel" && \
    mkdir -p "${MODEL_DIR}" && \
    curl --location --fail --show-error --retry 3 \
      "https://huggingface.co/camenduru/openpose/resolve/${OPENPOSE_MODEL_REVISION}/pose_iter_584000.caffemodel?download=true" \
      --output "${MODEL_FILE}" && \
    echo "${OPENPOSE_MODEL_SHA256}  ${MODEL_FILE}" | sha256sum --check -

# Python dependency required by YoloPose_yarp_server.py
RUN /opt/venv/bin/python -m pip install --no-cache-dir \
    "ultralytics==8.4.102"

RUN apt autoremove && apt clean
RUN rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp/*

WORKDIR $SOURCE_FOLDER

COPY .env /tmp/.env

RUN . /tmp/.env && \
    groupadd -g ${MY_GID} ${USER} && \
    useradd -m -s /bin/bash -u ${MY_UID} -g ${MY_GID} ${USER} && \
    usermod -aG sudo ${USER} && \
    echo "%sudo ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sudo && \
    chmod 0440 /etc/sudoers.d/sudo

# COPY run_yarpserver.sh .
# COPY run_atis_bridge.sh .
# COPY run_movenet.sh .
