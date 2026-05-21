# kaggle-kernel-basis 運用ガイド

Kaggle Notebooks (Kaggle Kernel の "Save & Run All" バッチ実行) を使い捨て GPU ノードとして扱い、ローカルから直接 GPU 学習を起動する基盤。Vast.ai (`docs/plans/vast-ai-basis/`) / RunPod (`docs/plans/runpod-basis/`) と並走し、開発者が **無料 GPU 枠** (T4x2 / P100、週 30h) を活用する選択肢を持つ。正本は Git + DVC/S3、Kaggle Kernel はステートレス。

## 設計サマリ

| 項目 | 採用案 | 理由 |
|------|--------|------|
| 成果物配送 | A1: `kaggle kernels output` → ローカルで `dvc add` | Kaggle 側に AWS creds 不要、internet OFF 競技にも将来流用可 |
| コード配送 | B2: `bot/` を Kaggle Dataset (`<user>/orbit-wars-bot`) として upload、notebook 本体は薄い entrypoint | uv 不在 / `pip install -e` で完結、`kaggle datasets version` で snapshot 化 |
| CLI | `dev/kaggle-kernel` + `bot/src/kaggle_kernel/` | `dev/runpod` / `dev/vast` と並列構造 |
| 互換性 | `policy/weights.pt` / run dir / `RunMetadata` 共有、`kaggle_kernel_meta` field を追加 | 三 provider 横断で `promote` / `cost-report` 共通化 |

## 前提セットアップ

```bash
# 1) AWS profile (DVC remote 用) — 既存
dev/dvc setup

# 2) Kaggle API key を bot/.env に追加 (推奨)
# https://www.kaggle.com/settings → "Create New API Token" で kaggle.json をダウンロード
# その中身を bot/.env に反映:
#   KAGGLE_USERNAME=<your-username>
#   KAGGLE_KEY=<your-key>

# 3) (初回のみ) bot/ を Kaggle Dataset として upload
dev/kaggle-kernel dataset push --commit-sha "$(git rev-parse HEAD)"
# → <user>/orbit-wars-bot dataset が作成され、以降は train ごとに version up される

# 4) (interactive mode 用、初回のみ) infra/ apply で IAM 拡張
#    DVC IAM に kaggle_interactive/* prefix の Put/Get/Delete を許可する。
cd infra/environment/dev && terraform apply
#    + Kaggle Web UI → Add-ons → Secrets で AWS_ACCESS_KEY_ID /
#    AWS_SECRET_ACCESS_KEY を登録 (interactive kernel 側が S3 channel を使う)
```

## 1 サイクルの流れ

```bash
# A) feature ブランチで params.yaml / コードを変更
vim params.yaml
git add -A && git commit -m "tune lr"
git push origin feature/<name>

# B) Kaggle Kernel 起動 (dataset push + kernel push + status polling)
dev/kaggle-kernel train "$(git rev-parse HEAD)" --case case1 --accelerator gpu-t4x2 --watch
#   → bot/ snapshot を dataset の新 version として upload
#   → notebook 自動生成 → kernel push
#   → status polling (QUEUED → RUNNING → COMPLETE/ERROR)
#   → watch オプション付きの場合、完了で desktop 通知

# C) 完了後、artifact を pull
dev/kaggle-kernel pull <run_id> --case case1
#   → kaggle kernels output で /kaggle/working/runs/<run_id>/ を取得
#   → data/output/models/imitation/case1/runs/<run_id>/ に配置
#   → dvc add + dvc push で S3 にも同期
#   → run.json の中身が pretty-printed 表示

# D) ローカル評価
ORBIT_WARS_WEIGHTS=data/output/models/imitation/case1/runs/<run_id>/best.pt \
  uv run --directory bot python -m pipeline.imitation.case1.evaluation.eval_vs_baseline \
  --episodes 300 --seed 0

# E) 採用するなら canonical に昇格
dev/kaggle-kernel promote <run_id> --case case1 [--eval-results path/to/eval.json]
#   → policy/weights.pt にコピー、dvc commit、run.json status=adopted
#   → 表示された git status を確認して `git commit` + `git push` + PR 作成

# F) 月次 quota レポート
dev/kaggle-kernel cost-report --month 2026-05
# → docs/experiment/kaggle_kernel_cost_report_2026-05.md に出力
```

## 三基盤の使い分け

| 観点 | Vast.ai | RunPod | Kaggle Kernel |
|------|---------|--------|---------------|
| 価格 | 同 GPU で 20-30% 安い | やや高い | **無料** (週 30h GPU 枠) |
| Reliability | community で揺れあり | Secure Cloud 安定 | Kaggle インフラに準ずる |
| 時間上限 | なし | なし | **9h GPU / 12h CPU** |
| 同時実行 | 任意 | 任意 | **~5 kernel** |
| GPU 選択 | offer 検索 | offer 検索 | accelerator 二択 (t4x2 / p100) |
| volume | network volume CRUD | Secure 専用 + DC 拘束 | なし (Dataset で代替) |
| Self-destroy | `vastai destroy` | `runpodctl stop pod` | kernel は完了で自動停止 |
| 採用シーン | コスト優先、短時間 | 安定性優先、長め run | **コスト 0 で小規模 imitation case を回す** |

三基盤は共通の DVC remote / canonical weights / run dir scheme を使う。`run.json` の `vast_offer_snapshot` / `runpod_offer_snapshot` / `kaggle_kernel_meta` field でどの provider 経由か追跡可能。

## トラブルシューティング

### kernel が ERROR で終了する
`dev/kaggle-kernel logs <run_id>` で stdout/stderr を取得 (kernel 完了後のみ可)。よくある原因:
- Rust simulator wheel のビルド失敗 → dataset に manylinux wheel を同梱しているか確認 (`dev/kaggle-kernel dataset status` の files listing)。
- 9h 上限超過 → `--max-hours 8.5` で内部 timeout を仕込むか、case を小さくする。

### `dev/kaggle-kernel train` で "no GPU quota remaining"
週次 30h の GPU 枠を使い切っている。`dev/kaggle-kernel cost-report` で残量確認、リセットまで待つか Vast/RunPod に切り替え。

### `kaggle kernels output` で run.json が見つからない
kernel が COMPLETE せず ERROR の場合、artifact は output に保存されない。`logs` で原因確認 → 再 push。

### `pip install -e /kaggle/input/orbit-wars-bot/` が遅い
依存 wheel が大きいため初回 5-10 分かかる。Kaggle 側 cache を利用するためにも `accelerator` を固定して run することを推奨。

## 関連ドキュメント

- [`00-codebase-research.md`](00-codebase-research.md) — 既存 vast / runpod 実装の調査
- [`01-web-research.md`](01-web-research.md) — Kaggle API spec, GPU quota, 制約値
- [`02-requirements.md`](02-requirements.md) — 機能 / 非機能要件
- [`03-architecture.md`](03-architecture.md) — ディレクトリ層、データフロー
- [`04-steps.md`](04-steps.md) — Phase 別実装ステップ
- [`05-risks.md`](05-risks.md) — リスクと緩和
- [`06-testing.md`](06-testing.md) — テスト戦略 + e2e smoke
