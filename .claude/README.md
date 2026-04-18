# Everything Claude Code

[![Stars](https://img.shields.io/github/stars/affaan-m/everything-claude-code?style=flat)](https://github.com/affaan-m/everything-claude-code/stargazers)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Shell](https://img.shields.io/badge/-Shell-4EAA25?logo=gnu-bash&logoColor=white)
![TypeScript](https://img.shields.io/badge/-TypeScript-3178C6?logo=typescript&logoColor=white)
![Go](https://img.shields.io/badge/-Go-00ADD8?logo=go&logoColor=white)
![Markdown](https://img.shields.io/badge/-Markdown-000000?logo=markdown&logoColor=white)

**Anthropicハッカソン優勝者によるClaude Code設定の完全コレクション。**

10ヶ月以上にわたる実プロダクト開発での集中的な日常使用を通じて進化した、本番環境対応のエージェント、スキル、フック、コマンド、ルール、MCP設定。

---

## ガイド

このリポジトリはコードのみです。ガイドですべてを解説しています。

<table>
<tr>
<td width="50%">
<a href="https://x.com/affaanmustafa/status/2012378465664745795">
<img src="https://github.com/user-attachments/assets/1a471488-59cc-425b-8345-5245c7efbcef" alt="Everything Claude Code 簡易ガイド" />
</a>
</td>
<td width="50%">
<a href="https://x.com/affaanmustafa/status/2014040193557471352">
<img src="https://github.com/user-attachments/assets/c9ca43bc-b149-427f-b551-af6840c368f0" alt="Everything Claude Code 詳細ガイド" />
</a>
</td>
</tr>
<tr>
<td align="center"><b>簡易ガイド</b><br/>セットアップ、基礎、設計思想。<b>まずこちらを読んでください。</b></td>
<td align="center"><b>詳細ガイド</b><br/>トークン最適化、メモリ永続化、評価、並列化。</td>
</tr>
</table>

| トピック | 学べる内容 |
|---------|-----------|
| トークン最適化 | モデル選択、システムプロンプトの軽量化、バックグラウンドプロセス |
| メモリ永続化 | セッション間のコンテキストを自動保存・読み込みするフック |
| 継続学習 | セッションから再利用可能なスキルへパターンを自動抽出 |
| 検証ループ | チェックポイント vs 継続評価、採点タイプ、pass@k指標 |
| 並列化 | Gitワークツリー、カスケード方式、インスタンスのスケーリング判断 |
| サブエージェント連携 | コンテキスト問題、反復的取得パターン |

---

## クロスプラットフォーム対応

このプラグインは **Windows、macOS、Linux** に完全対応しています。すべてのフックとスクリプトは最大限の互換性のためにNode.jsで書き直されました。

### パッケージマネージャーの検出

プラグインは以下の優先順位で使用するパッケージマネージャー（npm、pnpm、yarn、bun）を自動検出します：

1. **環境変数**: `CLAUDE_PACKAGE_MANAGER`
2. **プロジェクト設定**: `.claude/package-manager.json`
3. **package.json**: `packageManager` フィールド
4. **ロックファイル**: package-lock.json、yarn.lock、pnpm-lock.yaml、bun.lockb から検出
5. **グローバル設定**: `~/.claude/package-manager.json`
6. **フォールバック**: 最初に見つかったパッケージマネージャー

パッケージマネージャーの設定方法：

```bash
# 環境変数で設定
export CLAUDE_PACKAGE_MANAGER=pnpm

# グローバル設定で設定
node scripts/setup-package-manager.js --global pnpm

# プロジェクト設定で設定
node scripts/setup-package-manager.js --project bun

# 現在の設定を検出
node scripts/setup-package-manager.js --detect
```


---

## 内容

このリポジトリは **Claude Code プラグイン** です。直接インストールするか、コンポーネントを手動でコピーできます。

```
everything-claude-code/
|-- .claude-plugin/   # プラグインとマーケットプレイスのマニフェスト
|   |-- plugin.json         # プラグインのメタデータとコンポーネントパス
|   |-- marketplace.json    # /plugin marketplace add 用マーケットプレイスカタログ
|
|-- agents/           # 委任用の特化サブエージェント
|   |-- planner.md           # 機能実装の計画
|   |-- architect.md         # システム設計の判断
|   |-- tdd-guide.md         # テスト駆動開発
|   |-- code-reviewer.md     # 品質・セキュリティレビュー
|   |-- security-reviewer.md # 脆弱性分析
|   |-- build-error-resolver.md
|   |-- e2e-runner.md        # Playwright E2Eテスト
|   |-- refactor-cleaner.md  # デッドコード削除
|   |-- doc-updater.md       # ドキュメント同期
|   |-- python-reviewer.md       # Pythonコードレビュー
|   |-- python-build-resolver.md # Pythonビルドエラー解決
|
|-- skills/           # ワークフロー定義とドメイン知識
|   |-- coding-standards/           # 言語のベストプラクティス
|   |-- backend-patterns/           # API、データベース、キャッシュパターン
|   |-- frontend-patterns/          # React、Next.js パターン
|   |-- continuous-learning/        # セッションからパターンを自動抽出（詳細ガイド）
|   |-- continuous-learning-v2/     # 信頼度スコアリング付きインスティンクトベース学習
|   |-- iterative-retrieval/        # サブエージェント向け段階的コンテキスト改善
|   |-- strategic-compact/          # 手動コンパクション提案（詳細ガイド）
|   |-- tdd-workflow/               # TDD手法
|   |-- security-review/            # セキュリティチェックリスト
|   |-- eval-harness/               # 検証ループ評価（詳細ガイド）
|   |-- verification-loop/          # 継続的検証（詳細ガイド）
|   |-- python-patterns/            # Pythonイディオムとベストプラクティス
|   |-- python-testing/             # Pythonテストパターン（pytest、TDD）
|
|-- commands/         # クイック実行用スラッシュコマンド
|   |-- tdd.md              # /tdd - テスト駆動開発
|   |-- plan.md             # /plan - 実装計画
|   |-- playwright.md       # /playwright - E2Eテスト生成
|   |-- code-review.md      # /code-review - 品質レビュー
|   |-- typescript-lint.md    # /typescript-lint - ビルド・lintエラー修正
|   |-- remove-unused.md    # /remove-unused - デッドコード削除
|   |-- learn.md            # /learn - セッション中のパターン抽出（詳細ガイド）
|   |-- checkpoint.md       # /checkpoint - 検証状態の保存（詳細ガイド）
|   |-- verify.md           # /verify - 検証ループ実行（詳細ガイド）
|   |-- python-review.md     # /python-review - Pythonコードレビュー
|   |-- python-tdd.md        # /python-tdd - Python TDDワークフロー
|   |-- python-lint.md       # /python-lint - Pythonビルド・lintエラー修正
|   |-- skill-create.md     # /skill-create - git履歴からスキル生成（新規）
|   |-- instinct-status.md  # /instinct-status - 学習済みインスティンクト表示（新規）
|   |-- instinct-import.md  # /instinct-import - インスティンクトのインポート（新規）
|   |-- instinct-export.md  # /instinct-export - インスティンクトのエクスポート（新規）
|   |-- instinct-evolve.md  # /instinct-evolve - インスティンクトをスキルに統合（新規）
|
|-- rules/            # 常に従うガイドライン（~/.claude/rules/ にコピー）
|   |-- security.md         # 必須セキュリティチェック
|   |-- coding-style.md     # 不変性、ファイル構成
|   |-- testing.md          # TDD、80%カバレッジ要件
|   |-- git-workflow.md     # コミット形式、PRプロセス
|   |-- agents.md           # サブエージェントへの委任タイミング
|   |-- performance.md      # モデル選択、コンテキスト管理
|
|-- hooks/            # トリガーベースの自動化
|   |-- hooks.json                # 全フック設定（PreToolUse、PostToolUse、Stop など）
|   |-- memory-persistence/       # セッションライフサイクルフック（詳細ガイド）
|   |-- strategic-compact/        # コンパクション提案（詳細ガイド）
|
|-- scripts/          # クロスプラットフォーム Node.js スクリプト（新規）
|   |-- lib/                     # 共有ユーティリティ
|   |   |-- utils.js             # クロスプラットフォーム ファイル/パス/システムユーティリティ
|   |   |-- package-manager.js   # パッケージマネージャーの検出と選択
|   |-- hooks/                   # フック実装
|   |   |-- session-start.js     # セッション開始時のコンテキスト読み込み
|   |   |-- session-end.js       # セッション終了時の状態保存
|   |   |-- pre-compact.js       # コンパクション前の状態保存
|   |   |-- suggest-compact.js   # 戦略的コンパクション提案
|   |   |-- evaluate-session.js  # セッションからのパターン抽出
|   |-- setup-package-manager.js # 対話式PM設定
|
|-- tests/            # テストスイート（新規）
|   |-- lib/                     # ライブラリテスト
|   |-- hooks/                   # フックテスト
|   |-- run-all.js               # 全テスト実行
|
|-- contexts/         # 動的システムプロンプト注入コンテキスト（詳細ガイド）
|   |-- dev.md              # 開発モードコンテキスト
|   |-- review.md           # コードレビューモードコンテキスト
|   |-- research.md         # 調査・探索モードコンテキスト
|
|-- examples/         # 設定例とセッション例
|   |-- CLAUDE.md           # プロジェクトレベル設定例
|   |-- user-CLAUDE.md      # ユーザーレベル設定例
|
|-- mcp-configs/      # MCPサーバー設定
|   |-- mcp-servers.json    # GitHub、Supabase、Vercel、Railway など
|
|-- marketplace.json  # セルフホストマーケットプレイス設定（/plugin marketplace add 用）
```

---

## エコシステムツール

### スキルクリエイター

リポジトリからClaude Codeスキルを生成する2つの方法：

#### オプションA: ローカル分析（組み込み）

外部サービスなしでローカル分析を行う `/skill-create` コマンド：

```bash
/skill-create                    # 現在のリポジトリを分析
/skill-create --instincts        # 継続学習用のインスティンクトも生成
```

git履歴をローカルで分析し、SKILL.mdファイルを生成します。

#### オプションB: GitHub App（上級者向け）

高度な機能（10k+コミット、自動PR、チーム共有）向け：

[GitHub Appをインストール](https://github.com/apps/skill-creator) | [ecc.tools](https://ecc.tools)

```bash
# 任意のIssueにコメント:
/skill-creator analyze

# またはデフォルトブランチへのプッシュで自動トリガー
```

どちらのオプションでも以下が作成されます：
- **SKILL.mdファイル** - Claude Code用のすぐ使えるスキル
- **インスティンクトコレクション** - continuous-learning-v2用
- **パターン抽出** - コミット履歴から学習

### 継続学習 v2

インスティンクトベースの学習システムがパターンを自動学習します：

```bash
/instinct-status        # 信頼度付きの学習済みインスティンクトを表示
/instinct-import <file> # 他者からインスティンクトをインポート
/instinct-export        # インスティンクトを共有用にエクスポート
/instinct-evolve        # 関連インスティンクトをスキルに統合
```

詳細は `skills/continuous-learning-v2/` を参照してください。

---

## インストール

### オプション1: プラグインとしてインストール（推奨）

最も簡単な方法 - Claude Codeプラグインとしてインストール：

```bash
# このリポジトリをマーケットプレイスとして追加
/plugin marketplace add affaan-m/everything-claude-code

# プラグインをインストール
/plugin install everything-claude-code@everything-claude-code
```

または `~/.claude/settings.json` に直接追加：

```json
{
  "extraKnownMarketplaces": {
    "everything-claude-code": {
      "source": {
        "source": "github",
        "repo": "affaan-m/everything-claude-code"
      }
    }
  },
  "enabledPlugins": {
    "everything-claude-code@everything-claude-code": true
  }
}
```

これですべてのコマンド、エージェント、スキル、フックにアクセスできます。

> **注意:** Claude Codeプラグインシステムはプラグイン経由での `rules` 配布に対応していません（[上流の制限](https://code.claude.com/docs/en/plugins-reference)）。ルールは手動でインストールする必要があります：
>
> ```bash
> # まずリポジトリをクローン
> git clone https://github.com/affaan-m/everything-claude-code.git
>
> # オプションA: ユーザーレベルのルール（全プロジェクトに適用）
> cp -r everything-claude-code/rules/* ~/.claude/rules/
>
> # オプションB: プロジェクトレベルのルール（現在のプロジェクトのみに適用）
> mkdir -p .claude/rules
> cp -r everything-claude-code/rules/* .claude/rules/
> ```

---

### オプション2: 手動インストール

インストール内容を手動で管理したい場合：

```bash
# リポジトリをクローン
git clone https://github.com/affaan-m/everything-claude-code.git

# エージェントをClaude設定にコピー
cp everything-claude-code/agents/*.md ~/.claude/agents/

# ルールをコピー
cp everything-claude-code/rules/*.md ~/.claude/rules/

# コマンドをコピー
cp everything-claude-code/commands/*.md ~/.claude/commands/

# スキルをコピー
cp -r everything-claude-code/skills/* ~/.claude/skills/
```

#### settings.json にフックを追加

`hooks/hooks.json` のフックを `~/.claude/settings.json` にコピーしてください。

#### MCP の設定

`mcp-configs/mcp-servers.json` から必要なMCPサーバーを `~/.claude.json` にコピーしてください。

**重要:** `YOUR_*_HERE` プレースホルダーを実際のAPIキーに置き換えてください。

---

## 主要コンセプト

### エージェント

サブエージェントは限定されたスコープで委任されたタスクを処理します。例：

```markdown
---
name: code-reviewer
description: コードの品質、セキュリティ、保守性をレビュー
tools: ["Read", "Grep", "Glob", "Bash"]
model: opus
---

あなたはシニアコードレビュアーです...
```

### スキル

スキルはコマンドやエージェントから呼び出されるワークフロー定義です：

```markdown
# TDD ワークフロー

1. まずインターフェースを定義
2. 失敗するテストを書く（RED）
3. テストを通す最小限のコードを実装（GREEN）
4. リファクタリング（IMPROVE）
5. 80%以上のカバレッジを確認
```

### フック

フックはツールイベントで発火します。例 - console.log の警告：

```json
{
  "matcher": "tool == \"Edit\" && tool_input.file_path matches \"\\\\.(ts|tsx|js|jsx)$\"",
  "hooks": [{
    "type": "command",
    "command": "#!/bin/bash\ngrep -n 'console\\.log' \"$file_path\" && echo '[Hook] console.log を削除してください' >&2"
  }]
}
```

### ルール

ルールは常に従うガイドラインです。モジュール化して管理：

```
~/.claude/rules/
  security.md      # シークレットのハードコード禁止
  coding-style.md  # 不変性、ファイル制限
  testing.md       # TDD、カバレッジ要件
```

---

## テストの実行

プラグインには包括的なテストスイートが含まれています：

```bash
# 全テスト実行
node tests/run-all.js

# 個別テストファイルの実行
node tests/lib/utils.test.js
node tests/lib/package-manager.test.js
node tests/hooks/hooks.test.js
```

---

## コントリビューション

**コントリビューションを歓迎します。**

このリポジトリはコミュニティリソースとして公開しています。以下をお持ちの方はぜひご参加ください：
- 有用なエージェントやスキル
- 巧みなフック
- より良いMCP設定
- 改善されたルール

詳細は [CONTRIBUTING.md](CONTRIBUTING.md) をご覧ください。

### コントリビューションのアイデア

- 言語固有のスキル（Python、Rustパターン）- Go は対応済み!
- フレームワーク固有の設定（Django、Rails、Laravel）
- DevOpsエージェント（Kubernetes、Terraform、AWS）
- テスト戦略（各種フレームワーク）
- ドメイン固有の知識（ML、データエンジニアリング、モバイル）

---

## 背景

Claude Codeの実験的公開時から使用しています。2025年9月のAnthropic x Forum Venturesハッカソンで [@DRodriguezFX](https://x.com/DRodriguezFX) と共に [zenith.chat](https://zenith.chat) を構築して優勝しました - すべてClaude Codeで開発。

これらの設定は複数の本番アプリケーションで実戦テスト済みです。

---

## 重要な注意事項

### コンテキストウィンドウの管理

**重要:** すべてのMCPを同時に有効にしないでください。200kのコンテキストウィンドウが、有効なツールが多すぎると70kまで縮小する可能性があります。

目安：
- 20-30個のMCPを設定
- プロジェクトごとに有効化は10個未満
- アクティブなツールは80個未満

不要なものはプロジェクト設定の `disabledMcpServers` で無効化してください。

### カスタマイズ

これらの設定は私のワークフローに最適化されています。以下をお勧めします：
1. 共感できるものから始める
2. 自分の技術スタックに合わせて修正
3. 使わないものは削除
4. 自分のパターンを追加

---

## スター履歴

[![Star History Chart](https://api.star-history.com/svg?repos=affaan-m/everything-claude-code&type=Date)](https://star-history.com/#affaan-m/everything-claude-code&Date)

---

## リンク

- **簡易ガイド（まずこちら）:** [Everything Claude Code 簡易ガイド](https://x.com/affaanmustafa/status/2012378465664745795)
- **詳細ガイド（上級者向け）:** [Everything Claude Code 詳細ガイド](https://x.com/affaanmustafa/status/2014040193557471352)
- **フォロー:** [@affaanmustafa](https://x.com/affaanmustafa)
- **zenith.chat:** [zenith.chat](https://zenith.chat)

---

## ライセンス

MIT - 自由に使用、必要に応じて修正、可能であれば還元してください。

---

**役に立ったらスターしてください。両方のガイドを読んで、素晴らしいものを作りましょう。**
