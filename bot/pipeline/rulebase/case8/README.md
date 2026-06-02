# case8 — baseline_v8 (case4 + predict cache 速度最適化、採用版)

case4 (LB745 production) を base に `physics.predict_planet_position` を
NumPy → dict cache 化することで挙動完全等価のまま turn_p95 を削減した版。

## 経緯

このディレクトリは **複数の multi-step 最適化試行 (beam / PGS / NaïveMCTS / step guard / thrash filter / accumulate burst)** の
集約点として再構築された。具体的な経緯は `docs/experiment/rulebase/20260504_case8_multistep_optimization/` 配下に
iter ごとの記録 (iter1-iter11) がある。

最終的に **採用された改善は「predict cache 化」のみ** (元 case13、20260506 試行) で、
他の試行 (PGS / NaïveMCTS / beam / 各種フィルタ) は **0-53% で採用基準 +5pp に未達**、
heuristic 系探索は飽和したと結論。詳細は memory `project_heuristic_search_saturation` 参照。

## 採用戦略 (現在)

`bot/pipeline/rulebase/case8/baseline/core/physics.py:predict_planet_position` で
module-level dict cache を用いて turn 内の重複計算を排除。
`bot/pipeline/rulebase/case8/baseline/agent.py` の `agent(obs)` 先頭で
`reset_predict_cache()` を呼び leak を防止。

### 仮説と検証

- **仮説**: pure function なので caching は数値的同一性を保ち、勝率不変・turn_p95 削減
- **検証 (200戦 vs baseline_v4)**:
  - 勝率 50.5% (101/200) — 採用条件 ±2pp 以内 ✅
  - turn_p95 ~0.59s (case4 ~0.79s, -25%) ✅
  - case4 timeouts 4 件 vs case8 0 件 → margin 効果裏付け
- **cProfile**: predict_planet_position tottime 39.1s → 5.6s (-86%)

## 構造

case4 と同型。差分は:
- `baseline/core/physics.py` に `_PREDICT_PLANET_CACHE` (module-level dict) と
  `reset_predict_cache()` を追加
- `baseline/agent.py` の `agent(obs)` 先頭で `reset_predict_cache()` を呼び出し

## Fully-JAX エージェント (`baseline_jax/`)

case8 (= baseline_v8) の決定パイプラインを **忠実に** fixed-shape JAX へ移植した版。
近似 (`case1/baseline_jax*` の per-source argmax) ではなく **本物のアルゴリズム**
(mission scoring → `argsort(-score)` → 逐次 greedy global allocator) を `lax.scan` で
再現し、`jax.vmap` でゲーム間を並列実行して GPU 上で高速に self-play / データ生成 /
学習相手として回せる。元は `feature/jax-rulebase-agent` ブランチの `case_jax` で、case8 に集約した。

| パス | 役割 |
|------|------|
| `baseline/` | case8 Python agent (上記)。**JAX parity の正解 (oracle)** を兼ねる |
| `baseline_jax/` | JAX 本体。`geometry/physics/aim` は case2 から複製 (parity 済)、scoring/missions/allocator/movements は新規。entry は `agent_jax.compute_actions` |
| `_bench/selfplay_gpu/` | vmap self-play の GPU throughput + 勝率 sanity bench |

- **強さゲート**: `compute_actions` (JAX) vs `baseline_v8` (Python) を 300 戦自己対戦し
  勝率 45-55% (Wilson CI が 50% を含む)。
- **速度ゲート**: vmap JAX self-play が CPU per-turn 版に対し明確な throughput 向上。
- **Kaggle submit 対象外**: 本体 (`baseline_jax/` / `_bench/`) は学習・評価・データ生成専用で、
  `pipeline/.submitignore` で archive から除外。Kaggle 提出は従来通り `main.py` → `baseline/` が担う。
- parity test: `tests/unit/pipeline/rulebase/case8/test_*_jax_parity.py`、
  vmap self-play e2e: `tests/e2e/pipeline/rulebase/case8/test_selfplay_jax.py`。

## 関連 docs

- `docs/experiment/rulebase/20260504_case8_multistep_optimization/` — 全 iter 記録
  - iter1-iter11: 不採用試行 (PGS / NaïveMCTS / beam / 各種フィルタ)
  - **iter12: 採用された predict cache の plan/result** (元 case13、集約時に iter12 として配置)

## 関連 memory

- `project_case8_predict_cache_adopted` — 200戦結果、cProfile、実装ポイント
- `project_heuristic_search_saturation` — 11 連敗の経緯
- `project_case4_turn_p95_at_limit` — actTimeout 上限張り付きの初期所見
- `project_case4_hot_path` — physics.py に CPU 集中の profiling 結果
