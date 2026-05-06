# case13 — baseline_v13 (case4 速度最適化)

case4 (LB745) の **挙動完全等価** を保ちつつ `physics.predict_planet_position` を NumPy precompute → dict lookup に置換した速度最適化版。後続 value head 注入のための turn budget 確保が目的。

## 仮説 / 採用条件

- 勝率変化 ≤ ±2pp (200戦 vs baseline_v4)
- turn_p95 ≤ 0.5s (case4 比 -25% 以上)
- 両条件達成で採用、value head 注入 plan の base にする

## 構造

case4 と同型。違いは:
- `baseline/core/world_model.py` に `predicted_planet_pos: dict[(planet_id, turn), (x, y)]` cache を追加
- `baseline/core/physics.py:predict_planet_position` が cache lookup を優先 (cache miss 時は元実装にフォールバック)

## 関連

- plan: `docs/experiment/rulebase/20260506_case13_predict_cache/plan.md`
- profiling 元: `project_case4_hot_path` memory
