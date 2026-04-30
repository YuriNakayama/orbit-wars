# Vast.ai 学習ノード用 base image.
# 目的: uv sync (CUDA wheels 3GB ダウンロード ~12 分) を image layer に焼き込み、
# Vast 起動時の uv sync を cache hit (~30秒) に短縮する.
#
# Build context は repo root を想定.
#   docker build -t <ecr-url>:latest -f infra/runtime/Dockerfile .

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# DVC remote (S3) を変えるとき build_and_push.sh から --build-arg で渡す.
ARG DVC_S3_BUCKET=orbit-wars-dvc-286854171013
ARG DVC_S3_PREFIX=remote/files/md5

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
# DVC remote の S3 prefix から直接 `aws s3 sync` でコンテンツアドレス cache を
# 取得することで、git/dvc init が不要になり build を簡素化する.
# build 時に AWS credentials を BuildKit secret 経由で受け取り、layer には残さない.
WORKDIR /opt/orbit-wars
# BuildKit secret 経由 (ローカル dev) または env 経由 (CodeBuild) の AWS auth に
# 両対応する. CodeBuild の場合は task role が credentials を提供するので
# secret mount は no-op (空ファイル) でも問題なく動く.
RUN --mount=type=secret,id=aws_creds,target=/root/.aws/credentials,required=false \
    pip install --no-cache-dir awscli==1.40.* >/dev/null && \
    mkdir -p .dvc/cache/files/md5 && \
    aws s3 sync \
        "s3://${DVC_S3_BUCKET}/${DVC_S3_PREFIX}" \
        .dvc/cache/files/md5 \
        --no-progress \
    && pip uninstall -y awscli >/dev/null

# Vast onstart は clone 先 (/workspace/orbit-wars) で `.dvc/cache` を
# /opt/orbit-wars/.dvc/cache に symlink して焼込み済 cache を流用する.

# clone 先 (onstart で /workspace/orbit-wars に clone される)
WORKDIR /workspace
