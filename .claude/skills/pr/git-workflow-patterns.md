---
name: Git Workflow
description: Git flow based PR creation and release management
version: 1.0.0
---

# Git Workflow

Git Flow に基づく PR 作成・リリース管理。

## Commands

- `/pr` - PR 作成フロー（ブランチ戦略に応じたマージターゲット自動選択）
- `/release-tag` - リリースタグ作成

## Branch Strategy

```
feature/*    → develop
develop      → release/v*.*.*
release/*    → main
hotfix/*     → main
```
