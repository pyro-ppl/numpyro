# Experimental Dockerfile for CUDA-enabled numpyro
# This image should be suitable for numpyro developers
# it installs the latest version of numpyro from git
# and includes [dev] for libraries needed for development
# Author/Maintainer: AndrewCSQ (web_enquiry at andrewchia dot tech)

FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# declare the image name
# note that this image uses Python 3.8
ENV IMG_NAME=11.2.2-cudnn8-devel-ubuntu20.04

# install python3 and pip on top of the base Ubuntu image
# unlike for release, we need to install git and setuptools too
# one would probably want build-essential (gcc and friends) as well
RUN apt update && \
    apt install python3-dev python3-pip git build-essential -y

# add .local/bin to PATH for tqdm and f2py
ENV PATH=/root/.local/bin:$PATH

# install python packages via pip
RUN pip3 install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# clone the numpyro git repository and run pip install
RUN git clone https://github.com/pyro-ppl/numpyro.git && \
    cd numpyro && \
    pip3 install -e .[dev, test]  # contains additional dependencies for NumPyro development
