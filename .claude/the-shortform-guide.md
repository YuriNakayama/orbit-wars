# Everything Claude Code 簡易ガイド

![ヘッダー: Anthropicハッカソン優勝者 - Claude Codeのヒントとコツ](./assets/images/shortform/00-header.png)

---

**2月の実験的公開からClaude Codeを愛用しており、[@DRodriguezFX](https://x.com/DRodriguezFX) と共に [zenith.chat](https://zenith.chat) を構築して、Anthropic x Forum Venturesハッカソンで優勝しました - すべてClaude Codeで開発。**

10ヶ月間の日常使用で築いた完全セットアップ：スキル、フック、サブエージェント、MCP、プラグイン、そして実際に効果があったもの。

---

## スキルとコマンド

スキルはルールのように動作しますが、特定のスコープとワークフローに限定されます。特定のワークフローを実行する際のプロンプトの省略形です。

Opus 4.5での長いコーディングセッション後にデッドコードや不要な.mdファイルを整理したい場合は `/remove-unused` を実行。テストが必要なら `/tdd`、`/playwright`、`/test-coverage`。スキルにはコードマップも含めることができます - コンテキストを探索に消費せずにClaudeがコードベースを素早くナビゲートする方法です。

![コマンドチェーンを表示するターミナル](./assets/images/shortform/02-chaining-commands.jpeg)
*コマンドのチェーン実行*

コマンドはスラッシュコマンドで実行されるスキルです。重複しますが、保存場所が異なります：

- **スキル**: `~/.claude/skills/` - より広範なワークフロー定義
- **コマンド**: `~/.claude/commands/` - クイック実行可能なプロンプト

```bash
# スキル構造の例
~/.claude/skills/
  pmx-guidelines.md      # プロジェクト固有のパターン
  coding-standards.md    # 言語のベストプラクティス
  tdd-workflow/          # README.mdを含むマルチファイルスキル
  security-review/       # チェックリストベースのスキル
```

---

## フック

フックは特定のイベントで発火するトリガーベースの自動化です。スキルとは異なり、ツール呼び出しとライフサイクルイベントに限定されます。

**フックの種類：**

1. **PreToolUse** - ツール実行前（検証、リマインダー）
2. **PostToolUse** - ツール実行後（フォーマット、フィードバックループ）
3. **UserPromptSubmit** - メッセージ送信時
4. **Stop** - Claudeの応答完了時
5. **PreCompact** - コンテキストコンパクション前
6. **Notification** - 権限リクエスト

**例: 長時間実行コマンド前のtmuxリマインダー**

```json
{
  "PreToolUse": [
    {
      "matcher": "tool == \"Bash\" && tool_input.command matches \"(npm|pnpm|yarn|cargo|pytest)\"",
      "hooks": [
        {
          "type": "command",
          "command": "if [ -z \"$TMUX\" ]; then echo '[Hook] セッション永続化のためtmuxの使用を検討してください' >&2; fi"
        }
      ]
    }
  ]
}
```

![PostToolUseフックのフィードバック](./assets/images/shortform/03-posttooluse-hook.png)
*PostToolUseフック実行時にClaude Codeで表示されるフィードバックの例*

**ヒント:** `hookify` プラグインを使えば、JSONを手書きする代わりに会話形式でフックを作成できます。`/hookify` を実行して、やりたいことを説明してください。

---

## サブエージェント

サブエージェントは、オーケストレーター（メインのClaude）が限定されたスコープでタスクを委任できるプロセスです。バックグラウンドまたはフォアグラウンドで実行でき、メインエージェントのコンテキストを解放します。

サブエージェントはスキルとうまく連携します - スキルのサブセットを実行できるサブエージェントにタスクを委任し、それらのスキルを自律的に使用させることができます。特定のツール権限でサンドボックス化することも可能です。

```bash
# サブエージェント構造の例
~/.claude/agents/
  planner.md           # 機能実装の計画
  architect.md         # システム設計の判断
  tdd-guide.md         # テスト駆動開発
  code-reviewer.md     # 品質・セキュリティレビュー
  security-reviewer.md # 脆弱性分析
  build-error-resolver.md
  e2e-runner.md
  refactor-cleaner.md
```

サブエージェントごとに許可するツール、MCP、権限を設定して適切にスコープを定義してください。

---

## ルールとメモリ

`.rules` フォルダにはClaudeが**常に**従うべきベストプラクティスの `.md` ファイルを配置します。2つのアプローチ：

1. **単一のCLAUDE.md** - すべてを1ファイルに（ユーザーまたはプロジェクトレベル）
2. **ルールフォルダ** - 関心事ごとにグループ化されたモジュール式 `.md` ファイル

```bash
~/.claude/rules/
  security.md      # シークレットのハードコード禁止、入力検証
  coding-style.md  # 不変性、ファイル構成
  testing.md       # TDDワークフロー、80%カバレッジ
  git-workflow.md  # コミット形式、PRプロセス
  agents.md        # サブエージェントへの委任タイミング
  performance.md   # モデル選択、コンテキスト管理
```

**ルールの例：**

- コードベースに絵文字を使わない
- フロントエンドで紫色系を避ける
- デプロイ前に必ずテストを実行
- 巨大ファイルよりモジュール式コードを優先
- console.logをコミットしない

---

## MCP（Model Context Protocol）

MCPはClaudeを外部サービスに直接接続します。APIの代替ではなく、プロンプト駆動のラッパーであり、情報のナビゲーションにおいてより柔軟性を提供します。

**例:** Supabase MCPを使えば、Claudeが特定のデータを取得し、コピー&ペーストなしで直接SQLを実行できます。データベース、デプロイメントプラットフォームなども同様です。

![Supabase MCPがテーブルを一覧表示](./assets/images/shortform/04-supabase-mcp.jpeg)
*Supabase MCPがpublicスキーマ内のテーブルを一覧表示する例*

**Claude内のChrome:** Claude が自律的にブラウザを操作し、クリックして動作を確認できる組み込みプラグインMCPです。

**重要: コンテキストウィンドウの管理**

MCPの選択は慎重に。すべてのMCPをユーザー設定に保持しつつ、**未使用のものはすべて無効化**しています。`/plugins` に移動してスクロールするか、`/mcp` を実行してください。

![/pluginsインターフェース](./assets/images/shortform/05-plugins-interface.jpeg)
*/pluginsを使用してMCPに移動し、現在インストールされているものとそのステータスを確認*

コンパクション前の200kコンテキストウィンドウが、有効なツールが多すぎると70kにまで縮小する可能性があります。パフォーマンスが大幅に低下します。

**目安:** 設定に20-30個のMCPを持ちつつ、有効化は10個未満 / アクティブなツールは80個未満に。

```bash
# 有効なMCPを確認
/mcp

# ~/.claude.json の projects.disabledMcpServers で未使用のものを無効化
```

---

## プラグイン

プラグインは、面倒な手動セットアップの代わりに簡単にインストールできるようツールをパッケージ化したものです。プラグインはスキル + MCPの組み合わせや、フック/ツールのバンドルにできます。

**プラグインのインストール：**

```bash
# マーケットプレイスを追加
claude plugin marketplace add https://github.com/mixedbread-ai/mgrep

# Claudeを開き、/pluginsを実行、新しいマーケットプレイスを見つけてインストール
```

![mgrepを表示するマーケットプレイスタブ](./assets/images/shortform/06-marketplaces-mgrep.jpeg)
*新しくインストールされたMixedbread-Grepマーケットプレイスの表示*

**LSPプラグイン** はエディタ外でClaude Codeを頻繁に実行する場合に特に便利です。Language Server Protocolにより、IDEを開かなくてもClaudeにリアルタイムの型チェック、定義へのジャンプ、インテリジェントな補完機能を提供します。

```bash
# 有効なプラグインの例
typescript-lsp@claude-plugins-official  # TypeScriptインテリジェンス
pyright-lsp@claude-plugins-official     # Python型チェック
hookify@claude-plugins-official         # 会話形式でフック作成
mgrep@Mixedbread-Grep                   # ripgrepより優れた検索
```

MCPと同じ注意 - コンテキストウィンドウに注意してください。

---

## ヒントとコツ

### キーボードショートカット

- `Ctrl+U` - 行全体を削除（バックスペース連打より速い）
- `!` - クイックbashコマンドプレフィックス
- `@` - ファイル検索
- `/` - スラッシュコマンドの開始
- `Shift+Enter` - 複数行入力
- `Tab` - 思考表示の切り替え
- `Esc Esc` - Claudeの中断 / コードの復元

### 並列ワークフロー

- **フォーク** (`/fork`) - 重複しないタスクを並列実行するために会話をフォーク（キューに入れたメッセージを連投する代わりに）
- **Gitワークツリー** - 重複する並列Claudeをコンフリクトなしで実行。各ワークツリーは独立したチェックアウト

```bash
git worktree add ../feature-branch feature-branch
# 各ワークツリーで別々のClaudeインスタンスを実行
```

### 長時間実行コマンド用のtmux

Claudeが実行するログ/bashプロセスをストリーム・監視：

https://github.com/user-attachments/assets/shortform/07-tmux-video.mp4

```bash
tmux new -s dev
# Claudeがここでコマンドを実行、デタッチ・リアタッチ可能
tmux attach -t dev
```

### mgrep > grep

`mgrep` はripgrep/grepからの大幅な改善です。プラグインマーケットプレイスからインストールし、`/mgrep` スキルを使用。ローカル検索とウェブ検索の両方に対応。

```bash
mgrep "function handleSubmit"  # ローカル検索
mgrep --web "Next.js 15 app router changes"  # ウェブ検索
```

### その他の便利なコマンド

- `/rewind` - 以前の状態に戻る
- `/statusline` - ブランチ、コンテキスト%、TODOでカスタマイズ
- `/checkpoints` - ファイルレベルの取り消しポイント
- `/compact` - 手動でコンテキストコンパクションを実行

### GitHub Actions CI/CD

GitHub ActionsでPRのコードレビューを設定。設定すればClaudeが自動的にPRをレビューできます。

![Claude botがPRを承認](./assets/images/shortform/08-github-pr-review.jpeg)
*Claudeがバグ修正PRを承認*

### サンドボックス

リスクのある操作にはサンドボックスモードを使用 - Claudeが実際のシステムに影響を与えない制限された環境で実行します。

---

## エディタについて

エディタの選択はClaude Codeのワークフローに大きく影響します。Claude Codeはどのターミナルからでも動作しますが、高機能なエディタと組み合わせることで、リアルタイムのファイル追跡、クイックナビゲーション、統合されたコマンド実行が可能になります。

### Zed（私のお気に入り）

[Zed](https://zed.dev) を使用しています - Rustで書かれているため、本当に高速です。瞬時に開き、巨大なコードベースも問題なく処理し、システムリソースをほとんど消費しません。

**Zed + Claude Codeが優れた組み合わせである理由：**

- **速度** - Rustベースのパフォーマンスにより、Claudeがファイルを高速編集しても遅延なし。エディタが追従できます
- **エージェントパネル統合** - ZedのClaude統合でClaude編集時のファイル変更をリアルタイム追跡。Claudeが参照するファイル間をエディタを離れずにジャンプ
- **CMD+Shift+Rコマンドパレット** - カスタムスラッシュコマンド、デバッガー、ビルドスクリプトへの検索可能なUIでのクイックアクセス
- **最小限のリソース使用** - 重い操作中にClaudeとRAM/CPUを取り合いません。Opus実行時に重要
- **Vimモード** - お好みならフルVimキーバインディング

![カスタムコマンド付きZedエディタ](./assets/images/shortform/09-zed-editor.jpeg)
*CMD+Shift+Rでカスタムコマンドドロップダウンを表示するZedエディタ。右下にブルズアイとしてフォローモードを表示。*

**エディタ非依存のヒント：**

1. **画面分割** - 片側にClaude Codeのターミナル、もう片側にエディタ
2. **Ctrl + G** - Claudeが現在作業中のファイルをZedですぐに開く
3. **自動保存** - 自動保存を有効にしてClaudeのファイル読み取りが常に最新に
4. **Git統合** - エディタのGit機能でコミット前にClaudeの変更をレビュー
5. **ファイルウォッチャー** - ほとんどのエディタは変更されたファイルを自動リロード、有効であることを確認

### VSCode / Cursor

これも有効な選択肢でClaude Codeとうまく動作します。ターミナル形式で使用し、`\ide` でエディタとの自動同期によるLSP機能を有効化（プラグインにより多少冗長）。または、エディタとより統合されたUIを持つ拡張機能を選択できます。

![VS Code Claude Code拡張機能](./assets/images/shortform/10-vscode-extension.jpeg)
*VS Code拡張機能はClaude Codeのネイティブグラフィカルインターフェースを提供し、IDEに直接統合されます。*

---

## 私のセットアップ

### プラグイン

**インストール済み：**（通常はこのうち4-5個のみ有効化）

```markdown
ralph-wiggum@claude-code-plugins       # ループ自動化
frontend-design@claude-code-plugins    # UI/UXパターン
commit-commands@claude-code-plugins    # Gitワークフロー
security-guidance@claude-code-plugins  # セキュリティチェック
pr-review-toolkit@claude-code-plugins  # PR自動化
typescript-lsp@claude-plugins-official # TSインテリジェンス
hookify@claude-plugins-official        # フック作成
code-simplifier@claude-plugins-official
feature-dev@claude-code-plugins
explanatory-output-style@claude-code-plugins
code-review@claude-code-plugins
context7@claude-plugins-official       # ライブドキュメント
pyright-lsp@claude-plugins-official    # Python型
mgrep@Mixedbread-Grep                  # より良い検索
```

### MCPサーバー

**設定済み（ユーザーレベル）：**

```json
{
  "github": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"] },
  "firecrawl": { "command": "npx", "args": ["-y", "firecrawl-mcp"] },
  "supabase": {
    "command": "npx",
    "args": ["-y", "@supabase/mcp-server-supabase@latest", "--project-ref=YOUR_REF"]
  },
  "memory": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"] },
  "sequential-thinking": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
  },
  "vercel": { "type": "http", "url": "https://mcp.vercel.com" },
  "railway": { "command": "npx", "args": ["-y", "@railway/mcp-server"] },
  "cloudflare-docs": { "type": "http", "url": "https://docs.mcp.cloudflare.com/mcp" },
  "cloudflare-workers-bindings": {
    "type": "http",
    "url": "https://bindings.mcp.cloudflare.com/mcp"
  },
  "clickhouse": { "type": "http", "url": "https://mcp.clickhouse.cloud/mcp" },
  "AbletonMCP": { "command": "uvx", "args": ["ableton-mcp"] },
  "magic": { "command": "npx", "args": ["-y", "@magicuidesign/mcp@latest"] }
}
```

ポイントはここ - 14個のMCPを設定していますが、プロジェクトごとに有効化は5-6個程度。コンテキストウィンドウを健全に保ちます。

### 主要フック

```json
{
  "PreToolUse": [
    { "matcher": "npm|pnpm|yarn|cargo|pytest", "hooks": ["tmuxリマインダー"] },
    { "matcher": "Write && .mdファイル", "hooks": ["README/CLAUDE以外はブロック"] },
    { "matcher": "git push", "hooks": ["レビュー用にエディタを開く"] }
  ],
  "PostToolUse": [
    { "matcher": "Edit && .ts/.tsx/.js/.jsx", "hooks": ["prettier --write"] },
    { "matcher": "Edit && .ts/.tsx", "hooks": ["tsc --noEmit"] },
    { "matcher": "Edit", "hooks": ["console.log警告のgrep"] }
  ],
  "Stop": [
    { "matcher": "*", "hooks": ["変更ファイルのconsole.logチェック"] }
  ]
}
```

### カスタムステータスライン

ユーザー、ディレクトリ、ダーティインジケーター付きgitブランチ、残りコンテキスト%、モデル、時刻、TODOカウントを表示：

![カスタムステータスライン](./assets/images/shortform/11-statusline.jpeg)
*Mac ルートディレクトリでのステータスラインの例*

```
affoon:~ ctx:65% Opus 4.5 19:52
▌▌ plan mode on (shift+tab to cycle)
```

### ルール構造

```
~/.claude/rules/
  security.md      # 必須セキュリティチェック
  coding-style.md  # 不変性、ファイルサイズ制限
  testing.md       # TDD、80%カバレッジ
  git-workflow.md  # コンベンショナルコミット
  agents.md        # サブエージェント委任ルール
  patterns.md      # APIレスポンス形式
  performance.md   # モデル選択（Haiku vs Sonnet vs Opus）
  hooks.md         # フックドキュメント
```

### サブエージェント

```
~/.claude/agents/
  planner.md           # 機能の分解
  architect.md         # システム設計
  tdd-guide.md         # テストファースト
  code-reviewer.md     # 品質レビュー
  security-reviewer.md # 脆弱性スキャン
  build-error-resolver.md
  e2e-runner.md        # Playwrightテスト
  refactor-cleaner.md  # デッドコード削除
  doc-updater.md       # ドキュメント同期
```

---

## 主なポイント

1. **複雑にしすぎない** - 設定はアーキテクチャではなくファインチューニングとして扱う
2. **コンテキストウィンドウは貴重** - 未使用のMCPとプラグインを無効化
3. **並列実行** - 会話をフォーク、gitワークツリーを使用
4. **繰り返しを自動化** - フォーマット、リント、リマインダーにフックを活用
5. **サブエージェントのスコープを限定** - ツールを限定 = 集中した実行

---

## 参考資料

- [プラグインリファレンス](https://code.claude.com/docs/en/plugins-reference)
- [フックドキュメント](https://code.claude.com/docs/en/hooks)
- [チェックポイント](https://code.claude.com/docs/en/checkpointing)
- [インタラクティブモード](https://code.claude.com/docs/en/interactive-mode)
- [メモリシステム](https://code.claude.com/docs/en/memory)
- [サブエージェント](https://code.claude.com/docs/en/sub-agents)
- [MCP概要](https://code.claude.com/docs/en/mcp-overview)

---

**注意:** これは詳細の一部です。高度なパターンについては[詳細ガイド](./the-longform-guide.md)を参照してください。

---

*NYCでのAnthropic x Forum Venturesハッカソンで [@DRodriguezFX](https://x.com/DRodriguezFX) と共に [zenith.chat](https://zenith.chat) を構築して優勝*
