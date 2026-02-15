FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && rm -f /etc/apt/sources.list.d/yarnpkg.list

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update && apt-get install -y nodejs \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code
