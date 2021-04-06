# Experimental Dockerfile for CUDA-enabled numpyro
# This image should be suitable for numpyro developers
# it installs the latest version of numpyro from git
# and includes [dev] for libraries needed for development
# Author/Maintainer: AndrewCSQ (web_enquiry at andrewchia dot tech)

FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# declare the image name
# note that this image uses Python 3.8
ENV IMG_NAME=11.2.2-cudnn8-devel-ubuntu20.04 \
    # declare the cuda version for pulling appropriate jaxlib wheel
    JAXLIB_CUDA=112

# install python3 and pip on top of the base Ubuntu image
# unlike for release, we need to install git and setuptools too
# one would probably want build-essential (gcc and friends) as well
RUN apt update && \
    apt install python3-dev python3-pip git build-essential -y

# add .local/bin to PATH for tqdm and f2py
ENV PATH=/root/.local/bin:$PATH

# install python packages via pip
# install pip-versions to detect the latest version of jax and jaxlib
RUN pip3 install pip-versions
# this uses latest version of jax and jaxlib available from pypi
RUN pip-versions latest jaxlib | xargs -I{} pip3 install jaxlib=={}+cuda${JAXLIB_CUDA} -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    jax

RUN pip3 install --user \
    jax==${JAX_VERSION} \
    # we pull wheels from google's api as per https://github.com/google/jax#installation
    # the pre-compiled wheels that google provides work for now. This may change in the future (and necessitate building from source)
    jaxlib==${JAXLIB_VERSION}+cuda${JAXLIB_CUDA} -f https://storage.googleapis.com/jax-releases/jax_releases.html

# clone the numpyro git repository and run pip install
RUN git clone https://github.com/pyro-ppl/numpyro.git && \
    cd numpyro && \
    pip3 install -e .[dev, test]  # contains additional dependencies for NumPyro development
