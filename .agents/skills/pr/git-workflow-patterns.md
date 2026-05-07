---
name: Git Workflow
description: GitHub flow based PR creation and release management
version: 2.0.0
---

# Git Workflow

GitHub flow に基づく PR 作成・リリース管理。`main` が唯一の長命ブランチで、すべての変更は短命のトピックブランチからの PR として `main` にマージされる。

## Commands

- `/pr` — PR 作成フロー（base は常に `main`）
- `/release-tag` — `main` の任意コミットから semver タグを打ってリリースを切る

## Branch Strategy

```
feature/<topic>  → main   # 新機能・改善
fix/<topic>      → main   # バグ修正・hotfix（緊急かどうかに関わらず fix）
```

ブランチ名は短く、PR タイトルから推測できる単語にする。

## Release Strategy

`develop` / `release/*` といった長命ブランチは持たない。リリースが必要になったら `main` の任意のコミット（通常 HEAD）に直接タグを打つ。詳細は `release-tag` skill を参照。

## Merge Style

- 既定は **squash merge**。`main` の履歴を線形に保ち、PR 単位で 1 コミットになるようにする
- マージ済みのトピックブランチは即削除（`gh pr merge --delete-branch` または GitHub UI）

## Conventions

- PR は小さく 1 トピックに絞る。複数の関心が混ざっていたら分割する
- レビュー前にローカルで CI 相当 (`dev/test-backend` 等) を通しておく
- コミット・PR タイトル・本文はすべて日本語（リポジトリ規約）
