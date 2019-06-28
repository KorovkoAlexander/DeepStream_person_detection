FROM 1a2w3d4r/deep_stream_cuda_tensorrt_dev:10.0

# Model installation

RUN apt install -y --no-install-recommends libopencv-dev

RUN apt install -y --no-install-recommends build-essential

WORKDIR /opt

RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-misc-headers-10-0_10.0.130-1_amd64.deb

RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-nvcc-10-0_10.0.130-1_amd64.deb

RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-curand-dev-10-0_10.0.130-1_amd64.deb

RUN yes | dpkg -i cuda-misc-headers-10-0_10.0.130-1_amd64.deb
RUN yes | dpkg -i cuda-nvcc-10-0_10.0.130-1_amd64.deb
RUN yes | dpkg -i cuda-curand-dev-10-0_10.0.130-1_amd64.deb

RUN apt install -y --no-install-recommends python3-dev

RUN pip3 install wheel
RUN pip3 install pycuda tqdm click

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

COPY ./model /model

COPY ./Openpose /Openpose
COPY ./Calib /Calib

WORKDIR /Openpose/nvdsinfer_openpose

RUN make

WORKDIR /Openpose/python

RUN chmod u+x run.sh

CMD ["./run.sh"]

