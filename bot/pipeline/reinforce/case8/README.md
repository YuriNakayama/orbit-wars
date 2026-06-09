# Reinforce Case7 — PFSP self-snapshot pool × case5 best support reward

`pipeline/reinforce/case8/` は **case6 の PFSP (Prioritized Fictitious
Self-Play) opponent 基盤** と **case5 で最良だった support (補助) reward** を
組み合わせたケース。狙いは、自己対戦で相手が強くなり続ける環境 (PFSP) に、
収束を速める ratio PBRS shaping を載せて両方の利得を同時に取ること。

- **opponent 基盤 = case6**: `noop → pool` curriculum。`pool` は **過去の自分の
  snapshot を FIFO で貯め** (`snapshot_every` ごとに現 model を push、`cap` 上限)、
  各 late iter で「pooled past-self snapshot」か `baseline_jax_full` を選ぶ。
  `priority: f_hard` 時は勝ちにくい相手 ((1−win_ema)^p) を優先サンプル。
- **support reward = case5 H4**: `shaping_mode: ratio` / `shaping_coef: 1.0`。
  保持割合 `Φ = mine/(mine+enemy)` を ship と planet 両方で PBRS 差分加算。
  case5 実験 (`docs/experiment/reinforce/20260527_case5_support_reward/`) で
  last-10 0.820 / trend +0.668 を出した最良構成。

case6 から graft したのは `training/rollout_jax.py` の shaping 部
(`_shaping_potentials` / `_shaping_coefs` + ratio/combined/ratio_prod + dense/
clip/time の knob) と、`training/train_jax.py` でそれらを config から
`collect_rollout_jax` へ通す配線のみ。opponent dispatch (`self_snapshot` /
`_OpponentPool` / `_PrioritizedOpponentSelector`) は case6 のまま。

## 対戦相手モード (rollout_jax.py)

| mode | 内容 |
|------|------|
| `noop` | 何も発射しない (curriculum early) |
| `baseline_jax_lite` / `baseline_jax_full` | JAX ルールベース |
| `self_snapshot` | 凍結 self snapshot を決定論で対戦 (pool の中身) |
| `python_v1` / `v4` / `v8` | 本物 Python baseline (pure_callback、重い) |

## support reward の knob (shaping)

| key | case8 既定 (local_combo) | 意味 |
|-----|------|------|
| `shaping_mode` | `ratio` | 保持割合 PBRS (case5 H4) |
| `shaping_coef` | `1.0` | ship/planet 比率 Δ の共通係数 |
| `coef_ship` / `coef_planet` | 0 | `combined` mode 用 |
| `shaping_clip` | 0 | per-turn shaping の band-clip (H7、既定 off) |
| `dense_coef_*` | 0 | 非PBRS dense (H3、PBRS破綻するので off) |
| `time_bonus_coef` / `time_penalty_coef` | 0 | 早期勝利bonus / 引き伸ばしpenalty (H6、既定 off) |

## ローカル ~20min 実験

```bash
cd bot
uv run python -m pipeline.reinforce.case8.training.train_jax \
  --config pipeline/reinforce/case8/configs/local_combo_20min.yaml
```

`local_combo_20min.yaml` = ratio/coef=1.0 + self_snapshot pool (f_hard)、
ep=8 × horizon=200 × 16 iter (laptop CPU 実測 ~71s/iter、self_snapshot は
rollout を重くしない)。`best.pt` は npz。

### 追加学習 (resume)

保存済み `best.pt` (npz) から続きを学習できる。config の
`training.resume_from: <path/to/best.pt>` を指定すると、`_build_model` →
(BC) → `_maybe_resume` の順で重みが上書きされ、続きから PPO が回る
(`_load_npz_into_model` は `_save_best_pt` の逆、round-trip bit 一致)。
KL anchor の BC reference は resume 前の BC 重みで固定される。

```bash
uv run python -m pipeline.reinforce.case8.training.train_jax \
  --config <your.yaml>   # training.resume_from: data/.../runs/<id>/best.pt
```

> 本番 (200 iter) は RunPod GPU で。PFSP 系は rollout が相手 forward 分だけ
> 重くなり得るため iterations を抑えるか uptime を手動監視すること
> (memory `project_reinforce_self_snapshot_cost`)。

## アーキテクチャ / 設計原則 / JAX 化

backbone・head・BC 互換・JAX 化 (PR #74) は case3/5/6 と同一。
詳細は `pipeline/reinforce/case6/README.md` の該当節を参照
(case8 は opponent 基盤を case6 から、shaping を case5 から継承)。

agent registry: `rl_v7` (`bot/src/dataset/selfplay/agents.py`)。

## 既知のリスク

- case6 PFSP は JAX 自己対戦で伸びても **本物 baseline_v1 戦で 0/10** だった
  (train(JAX近似rule)/eval(本物v1) ギャップ、memory `project_reinforce_case6_live_eval`)。
  case8 も live 300戦 + featurizer parity での検証が採否の前提。
