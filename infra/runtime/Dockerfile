# Vast.ai 学習ノード用 base image.
# 目的: uv sync (CUDA wheels 3GB ダウンロード ~12 分) を image layer に焼き込み、
# Vast 起動時の uv sync を cache hit (~30秒) に短縮する.
#
# Build context は repo root を想定.
#   docker build -t <ecr-url>:latest -f infra/runtime/Dockerfile .

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# git は Vast onstart の git clone で必須. その他は uv build に必要な最小限.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv インストール (system path に配置)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# 依存だけ先に layer に焼く. pyproject.toml / uv.lock の変更時のみ rebuild.
WORKDIR /opt/orbit-wars/backend
COPY backend/pyproject.toml backend/uv.lock ./

# uv sync は project 自身も install しようとするので --no-install-project で
# 依存のみ install. project 本体は onstart で git clone 後に install される.
RUN uv sync --locked --no-install-project --no-dev

# Vast 起動時の uv cache hit を最大化するため、cache をユーザ可視な場所に維持.
ENV UV_CACHE_DIR=/root/.cache/uv

# DVC tracked データを image に焼き込み、Vast 起動時の dvc_pull を skip する.
# 焼込み対象: kaggle replays (~370MB) + case3 parquet (~387MB) + case3 *.dvc メタ.
# build 時に AWS credentials を BuildKit secret 経由で受け取り、
# image layer には credentials は残さない (RUN --mount=type=secret).
WORKDIR /opt/orbit-wars
COPY .dvc/config .dvc/config
COPY data/lake/kaggle_episodes/matches.dvc data/lake/kaggle_episodes/matches.dvc
COPY data/mart/imitation/case3/train_phase2_fixed.parquet.dvc \
     data/mart/imitation/case3/train_phase2_fixed.parquet.dvc
COPY data/mart/imitation/case3/val_phase2_fixed.parquet.dvc \
     data/mart/imitation/case3/val_phase2_fixed.parquet.dvc
RUN --mount=type=secret,id=aws_creds,target=/root/.aws/credentials \
    cd backend && \
    uv run --no-sync dvc pull --project /opt/orbit-wars \
        ../data/lake/kaggle_episodes/matches.dvc \
        ../data/mart/imitation/case3/train_phase2_fixed.parquet.dvc \
        ../data/mart/imitation/case3/val_phase2_fixed.parquet.dvc

# clone 先 (onstart で /workspace/orbit-wars に clone される)
WORKDIR /workspace
