# runpod-basis 運用ガイド

RunPod を使い捨て GPU ノードとして扱い、ローカルから直接 GPU 学習を起動する基盤。Vast.ai 基盤 (`docs/plans/vast-ai-basis/`) と並走し、開発者が価格・availability・reliability に応じて使い分ける。正本は Git + DVC/S3、RunPod はステートレス。

## 前提セットアップ

```bash
# 1) AWS profile (DVC remote 用) — 既存の dvc-setup を一度だけ
dev/dvc setup

# 2) RunPod API key を bot/.env に追加
# https://runpod.io/console/user/settings で発行
echo "RUNPOD_API_KEY=<your-key>" >> bot/.env

# 3) (推奨) network volume を 1 個作成 — 立ち上げ時間を 5-15 分短縮
# 永続化対象: uv-cache / dvc-cache / data/lake (raw episodes) / data/mart (preprocessed)
# 300GB に設定: 学習データ追加 (kaggle_episodes / selfplay matches) と次期 RL モデルを見越した余裕
dev/runpod volume create orbit_wars --size 300 --data-center-id CA-MTL-3
# 以降は同じ DC で `dev/runpod train` を叩くと --volume-name で自動再利用される
# Volume の月額: $0.07/GB × 300 = ~$21/月
```

## 1 サイクルの流れ

```bash
# A) feature ブランチで params.yaml / コードを変更
vim params.yaml
git add -A && git commit -m "tune lr"
git push origin feature/<name>

# B) RunPod 起動
dev/runpod train "$(git rev-parse HEAD)" --case case1 --cloud-type SECURE
#   → get_gpus + get_gpu で上位 10 件が rich table で表示
#   → 番号入力で offer 選択
#   → 推定コストが --cost-limit (デフォルト $1.5) を超えたら確認プロンプト
#   → pod 起動後、pod_id と runpodctl pod logs <id> モニタコマンドが表示される

# C) onstart の進捗を別ターミナルで確認 (任意)
runpodctl pod logs <pod_id>

# D) onstart 完了後 (約 15-25 分)
dev/runpod pull <run_id> --case case1
#   → DVC pull で best.pt / metrics.json / run.json をローカルに復元
#   → run.json の中身が pretty-printed で表示される

# E) ローカル評価
ORBIT_WARS_WEIGHTS=data/output/models/imitation/case1/runs/<run_id>/best.pt \
  uv run --directory bot python -m pipeline.imitation.case1.evaluation.eval_vs_baseline \
  --episodes 300 --seed 0
# 結果を JSON にして dev/runpod promote に渡せば run.json に local_eval_results が記録される

# F) 採用するなら canonical に昇格
dev/runpod promote <run_id> --case case1 [--eval-results path/to/eval.json]
#   → policy/weights.pt にコピー、dvc commit、run.json status=adopted、git status 表示
#   → 表示された git status を確認して `git commit` + `git push` + PR 作成

# G) コスト確認 (今月分)
dev/runpod cost-report --month 2026-05 --case case1
# → docs/experiment/runpod_cost_report_<YYYY-MM>.md に出力
```

## Vast.ai 基盤との使い分け

| 観点 | Vast.ai (`dev/vast`) | RunPod (`dev/runpod`) |
|------|-----|-----|
| 価格 | 同 GPU で 20-30% 安い | やや高い |
| Reliability | community 系で揺れあり | Secure Cloud は T3/T4 DC |
| Volume | network volume CRUD 完備 | network volume は Secure 専用 + DC 拘束 |
| Self-destroy | `vastai destroy instance` (SDK 経由) | `runpodctl stop pod` (pod 内 pre-install) |
| 採用シーン | コスト優先、短時間学習 | 安定性優先、Secure Cloud + 長め run |

両基盤は共通の DVC remote / canonical weights / run dir scheme を使う。`run.json` の `vast_offer_snapshot` / `runpod_offer_snapshot` field でどちら経由か追跡可能。

## トラブルシューティング

### onstart が失敗して pod が残っている
trap で **失敗時は自動 destroy しない** 設計。`runpodctl pod logs <id>` で原因確認 → 必要なら `runpodctl pod ssh <id>` でログイン → 復旧不能なら `runpodctl pod stop <id>` (or Web UI から terminate)。

### `dev/runpod pull` で run.json が見つからない
- run dir が DVC remote に push されていない可能性。`runpodctl pod logs <id>` で `dvc push` が成功したか確認。
- `dvc.lock` がローカルに残っていない場合: `git pull` でブランチを最新化してから retry。

### `dev/runpod train` が "RUNPOD_API_KEY not found" と言う
`bot/.env` に `RUNPOD_API_KEY=...` を追加 (.env は git ignore 済み)。

### コストが想定より高い
- `--cost-limit 1.0` のように個別に下限を絞る。
- 実 dph は `runpodctl pod list` で確認可能。
- weekly に `dev/runpod cost-report` を実行して履歴を確認。

### Network volume の DC 不一致で pod が起動しない
Volume が乗っている DC で pod を立てる必要がある。`dev/runpod volume list` で DC を確認 → `--data-center-id <dc>` で pod の DC を揃える。`--cloud-type=COMMUNITY` に切替えると volume なしでも起動可 (キャッシュなし)。

### 2h タイムアウト保険
onstart 冒頭で `( sleep 7200 && runpodctl stop pod $RUNPOD_POD_ID ) &` を仕掛けている。trap が壊れても 2h で強制停止。学習が 2h を超える設計の場合は onstart テンプレを編集してタイムアウトを延長。

## 関連ドキュメント

- 機能要件: [`02-requirements.md`](02-requirements.md)
- アーキテクチャ: [`03-architecture.md`](03-architecture.md)
- 実装ステップ: [`04-steps.md`](04-steps.md)
- リスク: [`05-risks.md`](05-risks.md)
- テスト戦略: [`06-testing.md`](06-testing.md)
- 既存 Vast 基盤: [`../vast-ai-basis/README.md`](../vast-ai-basis/README.md)
