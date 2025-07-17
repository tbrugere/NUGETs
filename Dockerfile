ARG PYTHON_VERSION=3.11
# This is a relatively big image, if you need it slimer, here are a few possible tweaks
# 1. Create a build image based on cuda...-devel-... with git, base-devel, python3
# 2. Then copy all stuff (ie the code and .env) onto a cuda...-runtime-cudnn-... image
# 3. The runtime image doesn't need cuda-devel or any of the devel stuff.
# 4. Hell you could even remove setuptools.
# 5. And use a slimer base than ubuntu. eg ubi9
# See Dockerfile.slimmer for all these improvements, although crucially, i'm not sure that runs

# see https://containers-at-tacc.readthedocs.io/en/latest/singularity/03.mpi_and_gpus.html#apptainer-and-gpu-computing
# "As a base, we recommend starting with the official CUDA (nvidia/cuda) images from NVIDIA"
# can also be other tags, see https://hub.docker.com/r/nvidia/cuda
# using devel version because it contains nvcc which is needed to build torch-scatter
# fast-attn etc.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS before_apt
ENV CUDA_HOME=/usr/local/cuda

# Install uv 
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# speed up apt
RUN sed -i 's/htt[p|ps]:\/\/archive.ubuntu.com\/ubuntu\//mirror:\/\/mirrors.ubuntu.com\/mirrors.txt/g' /etc/apt/sources.list

FROM before_apt 

ARG DEBIAN_FRONTEND=noninteractive

# git and g++ are needed for build
# technically I don't think we need them on the final image,
# so a possible improvement could be to make a separate build image and copy the env
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -y && \
    apt-get install --no-install-recommends -y git g++ 

RUN uv python install $PYTHON_VERSION

WORKDIR /app
RUN mkdir -p /app/workdir

# Install dependencies
# fixes a bug where some packages that haven't been updated to recent versions
# of python build system need setuptools and/or torch to be installed so they can be built
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock,ro \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,ro \
    uv sync --locked --no-install-project --only-group build-tools

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock,ro \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,ro \
    uv sync --locked --no-install-project --no-build-isolation --no-dev 


# Copy the requirements file and code
# note: I am not copying docs and tests
COPY nugets nugets.sh CGAL config.yaml pyproject.toml uv.lock static_configs /app

# CMD ["./entrypoint.sh"]
CMD ./nugets.sh wandb_sweep $SWEEP_ID
