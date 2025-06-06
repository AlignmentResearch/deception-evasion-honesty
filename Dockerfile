ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
FROM ${BASE_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive

ARG UID=1001
ARG GID=1001
ARG USERNAME=dev

# Update the package list, install sudo and curl, create a non-root user, and grant password-less sudo permissions
RUN apt update \
    && apt install -y sudo curl \
    && addgroup --gid $GID ${USERNAME} \
    && adduser --uid $UID --gid $GID --disabled-password --gecos "" ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Devbox niceties
WORKDIR "/workspace"
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml ./
COPY --chown=${USERNAME}:${USERNAME} . .

# RUN apt-get update && \
#     apt-get upgrade -y && \
#     apt-get install -y tmux less rsync git 


# RUN apt-get install -y git-lfs vim ssh python3-venv wget g++
# RUN apt-get install -y zsh gcc

