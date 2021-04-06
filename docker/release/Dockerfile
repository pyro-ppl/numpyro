# Experimental Dockerfile for CUDA-enabled numpyro
# This image should be suitable for developers on top of numpyro, or for end-users
# who wish to hit the ground running with CUDA+numpyro.
# Author/Maintainer: AndrewCSQ (web_enquiry at andrewchia dot tech)

FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# declare the image name
# note that this image uses Python 3.8
ENV IMG_NAME=11.2.2-cudnn8-devel-ubuntu20.04 \
    # declare what jaxlib, jax, and numpyro versions to use
    # right now this is a manual process - in the future it should be automated
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    JAXLIB_CUDA=112 \
    JAXLIB_VERSION=0.1.62 \
    JAX_VERSION=0.2.10 \
    NUMPYRO_VERSION=0.6.0

# install python3 and pip on top of the base Ubuntu image
RUN apt update && \
    apt install python3-dev python3-pip -y

# add .local/bin to PATH for tqdm and f2py
ENV PATH=/root/.local/bin:$PATH

# install python packages via pip
RUN pip3 install --user \
    numpyro==${NUMPYRO_VERSION} \
    jax==${JAX_VERSION} \
    # we pull wheels from google's api as per https://github.com/google/jax#installation
    # the pre-compiled wheels that google provides work for now. This may change in the future (and necessitate building from source)
    jaxlib==${JAXLIB_VERSION}+cuda${JAXLIB_CUDA} -f https://storage.googleapis.com/jax-releases/jax_releases.html