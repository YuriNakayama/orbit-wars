# Kaggle 提出自動化フロー — 外部技術リサーチ

## 公式ドキュメント・参照情報

### Kaggle CLI (`kaggle` パッケージ)

- パッケージ: `kaggle` (PyPI) — 公式 CLI。`pip install kaggle` でインストール。
- 認証:
  - 推奨: `~/.kaggle/kaggle.json` (権限 600)
  - CI 向け: 環境変数 `KAGGLE_USERNAME`, `KAGGLE_KEY`
- 主要コマンド (案件に関連するもの):

| コマンド | 用途 |
|---------|------|
| `kaggle competitions list --group entered` | 参加済みコンペ一覧 |
| `kaggle competitions submit <comp> -f <file> -m "<msg>"` | 提出 |
| `kaggle competitions submissions <comp>` | 提出履歴 |
| `kaggle competitions episodes <SUBMISSION_ID>` | エピソード履歴 |
| `kaggle competitions replay <EPISODE_ID>` | リプレイ取得 |
| `kaggle competitions logs <EPISODE_ID> <index>` | エージェントログ |
| `kaggle competitions leaderboard orbit-wars -s` | リーダーボード |

### Simulation コンペの提出形式

- **単一ファイル**: `main.py` のみで完結（外部パッケージは Kaggle 実行環境で提供される kaggle-environments ランタイムに含まれる）
- **複数ファイル**: tar.gz にバンドル（ルート直下に `main.py` 必須）
- **エージェント制約**:
  - `actTimeout=1s` → 1ターン以内に `agent(obs)` が返す必要
  - `remainingOverageTime` をエピソード共有バジェットとして利用可
  - Kaggle ランタイムに pandas, numpy, torch 等はプリインストール済み（Orbit Warsカーネルイメージ基準）

### GitHub Actions で Kaggle CLI を動かす

一般的パターン:

```yaml
env:
  KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
  KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
steps:
  - uses: actions/checkout@v5
  - uses: astral-sh/setup-uv@v6
  - run: uv sync --all-extras --dev
  - run: uv run kaggle competitions submit ...
```

環境変数で `~/.kaggle/kaggle.json` 不要。

## 類似 OSS プロジェクト

### 1. [Kaggle/kaggle-environments](https://github.com/Kaggle/kaggle-environments)

- 公式環境リポジトリ。 `kaggle_environments.make("orbit_wars", debug=True)` でローカル対戦を実行できる。
- リリースノートで 1.17.0 以降 orbit_wars が含まれている。
- **参考点**: `env.run([agent1, agent2, ...])` が標準API。agent の渡し方は file path も関数オブジェクトも可。

### 2. [Kaggle/docker-python](https://github.com/Kaggle/docker-python)

- Kaggle の Docker base image。ランタイム互換性チェックに利用可能（本タスクでは不要だが、依存追加時には参照）。

### 3. Lux AI / Halite 系の自動提出スクリプト

- 多くの参加者が「ローカル自己対戦 → 勝率がしきい値を超えたら自動提出」フローを作っている。
- **取り入れる点**: ローカル動作確認（エージェントを1回ドライランさせて例外を出さないこと）を必須ステップにする。

## パターン比較

| 観点 | 選択肢A: CLI 直呼び | 選択肢B: Python クライアント (`kaggle.api`) | 選択肢C: GitHub Actions 単独 |
|------|---------------------|------------------------------------------|-----------------------------|
| 学習コスト | 低 | 中 | 低 |
| ローカル開発体験 | ◎ | ○ | × (PRしないと動かない) |
| CI 統合 | ◎ | ◎ | ◎ |
| 出力パース | 面倒 | 簡単 (Python) | 同左 |
| 推奨 | **⭐ ローカルは CLI、Python側から subprocess** | - | - |

## ライブラリ選定

| 候補 | ✅ 利点 | ⚠️ 欠点 | メンテ状態 | 採用 |
|------|--------|---------|-----------|------|
| `kaggle` (公式) | 実績豊富、docs潤沢 | CLI がランタイムに常駐 | Active | ⭐ 採用 |
| 自前 REST 実装 | 依存減 | 保守コスト高 | - | 見送り |
| `typer` | 既に依存済、型安全 | - | Active | ⭐ 採用 |
| `rich` | 既に依存済、コンソール綺麗 | - | Active | ⭐ 採用 |

## API/プロトコル確認

- `kaggle competitions submit` は内部的に `POST /api/v1/competitions/submissions/submit/<comp>` を叩く。レスポンスは `status`, `submitted`, `publicScore` などを含む。
- レート制限: 1日5提出まで（Orbit Wars固有）。CLI 側でハンドリングされないので呼び出し側で履歴確認する必要がある（`kaggle competitions submissions orbit-wars` を事前に呼ぶ）。

## リサーチまとめ

- **採用アーキテクチャ**: ローカルは `uv run kaggle ...` を subprocess で呼ぶ方式。CI は `KAGGLE_USERNAME/KAGGLE_KEY` を GitHub Secrets に格納。
- **ドライラン必須**: `kaggle_environments.make` でエージェントを1ターン走らせて例外が出ないことを確認してから提出。
- **1日5提出制限への対応**: `--dry-run` フラグで「提出前のローカル検証のみ」を許可。実提出前に `kaggle competitions submissions` で今日の提出数を確認。
- **提出後の検証**: `kaggle competitions submissions` を一定時間ポーリング (`rich.progress`) して validation 成否をチェック。
