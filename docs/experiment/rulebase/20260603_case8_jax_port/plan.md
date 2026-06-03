# rulebase/case8 JAX full port — plan

> 記録: 2026-06-03 ~12:50 / 状態: in_progress (ループ cron, 10m)
> 親: case1 JAX 化完了 (docs/experiment/rulebase/20260603_case1_jax_port/)。
> user 指示「他のルールベースもJAX化して下さい」で scope を case2-9 へ拡張。

## 戦略: lineage 共有 core で全 case を効率化

case2-9 は全て case1/baseline からの fork (case0 は trivial 1-file で対象外)。
**case2,3,4,6,7,8,9 は共通アーキテクチャ** (lookahead + opponent_model + movements/
+ 新 mission capture/harass/swarm) を持つ「case2+ lineage」。case5 のみ別 shape
(agent_full.py / world_helpers.py)。

決定的事実 (diff 計測):
- **case4 vs case8 = 2 file diff のみ** (agent.py / core/physics.py)。lineage 内は近接。
- case8 core/: geometry **identical**, safety **identical**, world_model **44行**,
  physics 358行 (= t14 predict-cache, 挙動等価の perf 最適化), config 63行。
- → **case1 core_jax (geometry/safety/aim/physics/worldmodel) を土台に再利用可**。

最重要: case8 の **runtime feature flag**
- `OPPONENT_MODEL_ENABLED=False` → opponent_model.py (405 LOC) **実行されない**
- `LOOKAHEAD_ENABLED=False` → lookahead.py (141 LOC) **実行されない**
- `CRASH_EXPLOIT_ENABLED=True` / `HARASS_ENABLED=True` → これらは ON、要 port
- → 実 runtime 挙動 = `plan_moves(world)` + 新 mission/movements、OM/lookahead 抜き。
  port scope は raw 4226 LOC から ~550 LOC (OM+lookahead) を除外できる。

## 進め方 (case1 と同じ方法論、随所で高速+劣化なし確認)

1. **Step1 (このループ)**: 高速ローカル結合テスト確立 = JAX port vs **case8 Python**
   (jax 同士でなく書き換え前 Python と比較。loop 原則)。≤10min/game、数十対戦は避け
   tripwire (10-game ≥3 gate) で 0勝回避を最小検証。
2. **Step2**: case1 core_jax を case8 用に再利用/拡張する書換方針を検証 (geometry/safety
   は流用、worldmodel は 44行 delta、physics cache は挙動等価なので case1 physics_jax 流用)。
3. **実装**: 新 mission (harass/crash) + movements を JAX 化、parity test を bottom-up に。
   随所で tripwire + 速度確認。
4. case8 完了後、lineage delta (case2/3/4/6/7/9) と case5 を順次。

## 採否/制約 (skip list)

- Kaggle publicScore は引用しない。n<300 で結論を出さない (大規模 eval は避け tripwire 主)。
- OM/lookahead は default OFF につき **初期 port では実装しない** (flag OFF 挙動と一致すれば可)。
- 劣化回避 (≈0勝回避) が最優先。faithful byte-parity は case1 同様 win-rate と trade-off
  なら非追求。
- GPU/認証は不要、自走。
