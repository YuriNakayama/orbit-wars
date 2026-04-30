#!/usr/bin/env bash
# Build the Vast.ai runtime image and push to ECR.
#
# Prerequisites:
#   - Terraform applied (ECR repo + IAM user で credentials 取得済み)
#   - AWS credentials available via env / `--profile <name>`
#   - Docker daemon running (Docker Desktop or colima)
#
# Usage:
#   infra/runtime/build_and_push.sh [--tag <tag>] [--profile <aws-profile>]
#
# Tag policy:
#   - 'latest' tag は毎回上書き
#   - <commit-sha> tag を併存させて roll back 可能にする
set -euo pipefail

PROFILE="${AWS_PROFILE:-orbit-wars}"
TAG="latest"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

# Terraform output から ECR repo URL を取得 (terraform state を信頼).
ECR_URL=$(cd infra/environment/dev && terraform output -raw ecr_runtime_repository_url)
REGION=$(cd infra/environment/dev && terraform output -raw -json 2>/dev/null \
  | python3 -c 'import json, sys; d=json.load(sys.stdin); print(d.get("aws_region", {}).get("value", "ap-northeast-1"))' \
  || echo "ap-northeast-1")
REGISTRY="${ECR_URL%%/*}"
SHA=$(git rev-parse --short=7 HEAD)

echo "[runtime] ecr=${ECR_URL} region=${REGION} tag=${TAG} sha=${SHA}"

echo "[runtime] step=ecr_login"
aws ecr get-login-password --region "${REGION}" --profile "${PROFILE}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

echo "[runtime] step=docker_build"
# linux/amd64 を明示 (Apple Silicon でも Vast 側 amd64 が必要)
docker buildx build \
  --platform linux/amd64 \
  --file infra/runtime/Dockerfile \
  --tag "${ECR_URL}:${TAG}" \
  --tag "${ECR_URL}:${SHA}" \
  --load \
  .

echo "[runtime] step=docker_push"
docker push "${ECR_URL}:${TAG}"
docker push "${ECR_URL}:${SHA}"

echo "[runtime] done. usable image:"
echo "  ${ECR_URL}:${TAG}"
echo "  ${ECR_URL}:${SHA}"
