FROM ubuntu:24.04

ARG project=l2d
ARG username=yarikama

# Install minimal utilities and create user
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -md /home/${project} ${username}

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

USER ${username}
WORKDIR /home/${project}

# Compile bytecode for faster startup, copy mode for containers
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies only (mount cache for rebuild speed)
RUN --mount=type=cache,target=/home/${project}/.cache/uv,uid=1000 \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --frozen --no-dev --no-install-project

# Copy project source
COPY --chown=${username} . .
