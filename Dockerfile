FROM pytorch/pytorch as base

ARG UID=10000
ARG GID=101
ARG USERNAME=dev

ENV PYTHONUNBUFFERED=1 \
    # pip
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # Poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.7.1 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_PATH=/home \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_OPTIONS_SYSTEM_SITE_PACKAGES=true

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$PATH"

# Update the package list, install sudo, create a non-root user, and grant password-less sudo permissions
RUN apt update && \
    apt install -y sudo && \
    addgroup --gid $GID ${USERNAME} && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Install some useful packages
RUN apt update && \
    apt install -y rsync git vim

# Install Poetry
RUN apt install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 -

RUN chmod 777 /home

# Set the non-root user as the default user
USER ${USERNAME}
WORKDIR /home/${USERNAME}/auto-circuit

# Copying the Python project files
COPY --chown=${USERNAME}:${USERNAME} . /home/${USERNAME}/auto-circuit

# Install dependencies
RUN poetry install --with dev

WORKDIR /home
RUN rm -rf /home/${USERNAME}
