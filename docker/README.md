Docker Images for NumPyro
=========================

Experimental Dockerfiles for CUDA-accelerated NumPyro. There are two Dockerfiles included:

- `release`: intended for users of NumPyro. Includes the jax and jaxlib versions needed to run NumPyro only. (Right now, that is 0.2.10, 0.1.62, and 0.6.0 respectively).
- `dev`: intended for NumPyro developers. It includes the jax and jaxlib versions needed to run the latest release of NumPyro (same as above, for now), plus an installation of NumPyro from source.

## Pre-Requisites

The Docker host that the image is being deployed on will need to have the proprietary Nvidia driver as well as [the Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-docker) installed. OS support is constrained by what Nvidia supports on their toolkit. Right now, that means Linux only, although there is [experimental WSL2 support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#installing-wip), with an estimated ~30% hit to performance.

## Using the Dockerfiles

At the moment, these images are not distributed on Dockerhub or any similar Docker registry. Users must build the Docker images themselves. This can be done (from the root of the git repository) with the command

```
docker build -t <name_for_image:tag> docker/[dev or release]/.
```

The Docker image will then be available locally. For example, to open a shell in the Docker image, one would run

```
docker run -ti <name_for_image>
```

## Current State & Future Work

Design Choices:

- The Docker images do not include any users other than root, and do not include any other packages (such as Tensorflow or PyTorch). Users of the Docker image should layer their own requirements on top of these images.
- To avoid long build-times, the images use Google's provided CUDA wheels rather than building jaxlib from source.

Future Work:

- Right now the jax, jaxlib, and numpyro versions are manually specified, so they have to be updated every NumPyro release. There are two ways forward for this:
    1. If there is a CI/CD in place to build and push images to a repository like Dockerhub, then the jax, jaxlib, and numpyro versions can be passed in as environment variables (for example, if something like [Drone CI](http://plugins.drone.io/drone-plugins/drone-docker/) is used). If implemented this way, the jax/jaxlib/numpyro versions will be ephemereal (not stored in source code).
    2. Alternative, one can create a Python script that will modify the Dockerfiles upon release accordingly (using a hook of some sort). 
- For development work, it would be nice to be able to pull the latest versions of jax and jaxlib into the `dev` image. This would be made a lot easier if jax upstream published Docker images for CUDA-accelerated jaxlibs. There is [currently an issue for it](https://github.com/google/jax/issues/6340).
