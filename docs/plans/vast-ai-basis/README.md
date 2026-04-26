# vast-ai-basis 運用ガイド

Vast.ai を使い捨て GPU ノードとして扱い、ローカルから直接 GPU 学習を起動する基盤。
正本は Git + DVC/S3、Vast はステートレス。

## 前提セットアップ

```bash
# 1) AWS profile (DVC remote 用) — 既存の dvc-setup を一度だけ
dev/dvc-setup

# 2) Vast.ai API key を backend/.env に追加
cp backend/.env.example backend/.env
# → backend/.env を編集して VAST_API_KEY=<your-key> を入れる
# Key は https://cloud.vast.ai/manage-keys/ で発行
```

## 1 サイクルの流れ

```bash
# A) feature ブランチで params.yaml / コードを変更
vim params.yaml
git add -A && git commit -m "tune lr"
git push origin feature/<name>

# B) Vast 起動 (commit sha は git rev-parse HEAD で得られる完全 SHA でも短縮形でも可)
dev/vast-train "$(git rev-parse HEAD)"
#   → search offers の上位 10 件が rich table で表示される
#   → 番号入力で offer 選択
#   → 推定コストが --cost-limit (デフォルト $1.0) を超えたら確認プロンプト
#   → instance 起動後、run_id と vastai logs <id> モニタコマンドが表示される

# C) onstart の進捗を別ターミナルで確認 (任意)
vastai logs <instance_id>

# D) onstart 完了後 (約 15-25 分)
dev/vast-pull <run_id>
#   → DVC pull で best.pt / metrics.json / run.json をローカルに復元
#   → run.json の中身が pretty-printed で表示される

# E) ローカル評価 (例: case1 evaluator を run dir の weights で実行)
ORBIT_WARS_WEIGHTS=artifacts/models/imitation/case1/runs/<run_id>/best.pt \
  uv run --directory backend python -m pipeline.imitation.case1.evaluation.eval_vs_baseline \
  --episodes 300 --seed 0
# 結果を JSON にして dev/vast-promote に渡せば run.json に local_eval_results が記録される

# F) 採用するなら canonical に昇格
dev/vast-promote <run_id> [--eval-results path/to/eval.json]
#   → policy/weights.pt にコピー、dvc commit、run.json status=adopted、git status 表示
#   → 表示された git status を確認して `git commit` + `git push` + PR 作成

# G) コスト確認 (今月分)
dev/vast-cost-report
# → docs/experiment/vast_cost_report_<YYYY-MM>.md に出力
```

## トラブルシューティング

### onstart が失敗してインスタンスが残っている
trap で **失敗時は自動 destroy しない** 設計。`vastai logs <id>` で原因確認 → 必要なら `vastai ssh <id>` でログイン → 復旧不能なら手動で `vastai destroy instance <id>`。

### `dev/vast-pull` で run.json が見つからない
- run dir が DVC remote に push されていない可能性。`vastai logs <id>` で `dvc push` が成功したか確認。
- `dvc.lock` がローカルに残っていない場合: `git pull` でブランチを最新化してから retry。

### `dev/vast-train` が "VAST_API_KEY not found" と言う
`backend/.env` に `VAST_API_KEY=...` を追加。git ignore されているので commit はされない。

### コストが想定より高い
- `--cost-limit 0.5` のように個別に下限を絞る。
- 実 dph_total は `vastai show instances --raw` で確認可能。
- weekly に `dev/vast-cost-report` を実行して履歴を確認。

## 関連ドキュメント

- 機能要件: [`02-requirements.md`](02-requirements.md)
- アーキテクチャ: [`03-architecture.md`](03-architecture.md)
- 実装ステップ: [`04-steps.md`](04-steps.md)
- リスク: [`05-risks.md`](05-risks.md)
