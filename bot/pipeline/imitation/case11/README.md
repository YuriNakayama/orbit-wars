# Imitation Case11 — per_planet on refreshed Kaggle lake

case9 の per_planet variant のみを切り出した実験 case。最新の Kaggle episode
lake (~66k episodes) を winner-only / top80 でフィルタし、case9 の OOM 回避
上限 (`max_episodes=8000`) を外して全件で学習する。

## case9 per_planet との差分

| 項目 | case9 per_planet | case11 |
|------|------------------|--------|
| backbone | Set Transformer h=192 / ISAB×4 / heads=8 / inducing=24 | **同一** |
| featurizer | planet 41 / global 20 / candidate 8×14 | **同一** |
| head | per_planet 単一 + log1p(ships) | **同一** (`SUPPORTED_HEAD_MODES = ("per_planet",)` に制限) |
| データ | winners_only + top_team_rank=80 + `max_episodes=8000` | winners_only + top_team_rank=80 + **`max_episodes=null`** |
| mart path | `data/mart/imitation/case9_per_planet/` | `data/mart/imitation/case11_per_planet/` |
| weights_out | `policy/weights_per_planet.pt` | `policy/weights.pt` |

期待 mart 規模 (1v1 / draw 除外 / winner ∈ top80, 最新 lake 計測):

- ~5,042 対戦
- ~1.5M frame (loser_swap=true で counterfactual 2× 取り込み)
- train parquet ~10GB 想定。host RAM 64GB+ を確保 (case9 OOM 前科あり)。

## 手順

```bash
cd bot

# 1) 前処理 (winners_only + top80 + 1v1, 全件)
uv run python -m pipeline.imitation.case11.training.preprocess \
  --config pipeline/imitation/case11/configs/il_case11_per_planet.yaml

# 2) 学習 (45 epoch, batch=512, Set Transformer 192/8/24/×4)
uv run python -m pipeline.imitation.case11.training.train \
  --config pipeline/imitation/case11/configs/il_case11_per_planet.yaml
```

### RunPod (推奨)

```bash
git push origin <branch>
dev/runpod train <commit-sha> --case case11 --watch
dev/runpod pull <run_id> --case case11
dev/runpod promote <run_id> --case case11
```

## レジストリ

`src/dataset/selfplay/agents.py`:

```python
"il_v11_per_planet": "pipeline.imitation.case11.policy.agent:agent",
```

`src/runpod_io/config/cases.py` の `CASE_DEFAULTS["case11"]` で
preprocess + train を 1 コマンドにまとめている。

## 設計原則

- case 間独立。case9 からのコードコピーであり import 関係なし。
- head_mode は `per_planet` のみ (`SUPPORTED_HEAD_MODES = ("per_planet",)`)。
- canonical 重みは `policy/weights.pt` (DVC 管理、git untracked)。`dev/runpod promote` で上書き。
