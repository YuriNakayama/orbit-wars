---
name: CI/CD
description: GitHub Actions CI/CD pipeline monitoring, troubleshooting, and deployment for backend (ECS Fargate) and frontend (AWS Amplify)
activation: Checking CI/CD status, fixing pipeline failures, deploying to environments, or running E2E tests in CI
---

# CI/CD

GitHub Actions パイプラインの監視・トラブルシューティング・デプロイ。

## Commands

- `/ci-fix` - バックエンド・フロントエンドの CI 状況確認と失敗時の修正
- `/e2e-fix` - E2E テスト（Playwright）の CI 状況確認と失敗時の修正
- `/deploy` - バックエンド（ECS Fargate）・フロントエンド（AWS Amplify）のデプロイ実行

## Workflows

| ワークフロー | ファイル | トリガー | 内容 |
|-------------|---------|---------|------|
| CI Backend | `ci-backend.yml` | PR (backend/**) | ruff format/check, mypy, pytest |
| CI Frontend | `ci-frontend.yml` | PR (frontend/**) | type-check, format:check, lint, Playwright |
| CD Backend | `cd-backend.yml` | push main/release/develop | Docker build → ECR push → ECS deploy |
| CD Frontend | `cd-frontend.yml` | push release/** | Amplify deployment |
| E2E | `e2e.yml` | PR, workflow_dispatch | Playwright E2E against live environment |

## Environment Detection

| ブランチ | 環境 |
|---------|------|
| `main` | prod |
| `release/*` | stg |
| `develop` | dev |

## Infrastructure

- **Backend**: ECS Fargate (Docker, Python 3.12, FastAPI)
- **Frontend**: AWS Amplify (Next.js)
- **Registry**: Amazon ECR
- **Auth**: GitHub OIDC → AWS IAM
- **Secrets**: AWS SSM Parameter Store + KMS
- **Monitoring**: CloudWatch
