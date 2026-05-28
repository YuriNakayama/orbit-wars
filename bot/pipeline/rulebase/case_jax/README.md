# rulebase/case_jax — Fully-JAX rule-based agent

**目的**: case8 (= production champion baseline_v4 + predict cache) の決定パイプラインを
**忠実に** fixed-shape JAX へ移植し、`jax.vmap` でゲーム間を並列実行して GPU 上で高速に
self-play / データ生成 / 学習相手として回せるようにする。

近似 (`case1/baseline_jax*` の per-source argmax) ではなく **本物のアルゴリズム**
(mission scoring → `argsort(-score)` → 逐次 greedy global allocator) を `lax.scan` で
再現するため、Python baseline と自己対戦して **同程度の勝率** を保つ。

## ディレクトリ

| パス | 役割 |
|------|------|
| `baseline/` | case8 Python agent のコピー。**parity の正解 (oracle)** + CPU self-play 用 |
| `baseline_jax/` | JAX 本体。`geometry/physics/aim` は case2 から複製 (parity 済)、scoring/missions/allocator/movements は新規 |
| `main.py` | oracle agent の薄い entry point (`agent(obs)`) |
| `evaluation/` | 強さ parity 評価 (typer wrapper, src/evaluate 経由) |
| `_bench/` | GPU throughput bench |

## 強さ / 速度ゲート

- **強さ**: `compute_actions_jax` (JAX) vs `baseline_v8` (Python) を 300 戦自己対戦し
  勝率 45-55% (Wilson CI が 50% を含む)。
- **速度**: vmap した JAX self-play が CPU per-turn 版に対し明確な throughput 向上。

## 非対象

- **Kaggle submit**: 本 case は学習/評価/データ生成用。Kaggle 提出は **case8 が担う** (本 case の
  `agent` は per-turn actTimeout 最適化を目的としない)。
- **OM / lookahead**: case8 既定 OFF (`OM_V2_ENABLED` / `LOOKAHEAD_ENABLED = False`) 前提。

実装計画: `/Users/user/.claude/plans/replicated-shimmying-orbit.md`
