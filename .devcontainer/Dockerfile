# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version (use hirsuite or bionic on local arm64/Apple Silicon): hirsute, focal, bionic
ARG VARIANT="hirsute"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends build-essential

ENV CONDA_DIR="/usr/local/bin/miniconda3"
ENV PATH="$CONDA_DIR/bin:$PATH"

RUN wget \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh \
      -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p $CONDA_DIR \
    && rm -f ~/miniconda.sh

COPY environment.yaml requirements.txt /tmp/conda-tmp/
RUN umask 0002 \
    && conda env update -n base -f /tmp/conda-tmp/environment.yaml \
    && rm -rf /tmp/conda-tmp
