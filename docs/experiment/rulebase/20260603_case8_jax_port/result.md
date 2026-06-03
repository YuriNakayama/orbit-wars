# rulebase/case8 JAX full port — result

> 記録: 2026-06-03 ~13:00 / 状態: in_progress (ループ cron, 10m)
> plan: ./plan.md / 親: ../20260603_case1_jax_port/

## 経過 1 — scope 確定 + Step1 (高速結合テスト) 確立

### scope 計測

- case2-9 は全て case1/baseline fork。case2,3,4,6,7,8,9 = 共通 lineage
  (lookahead+OM+movements+新mission)、case5 のみ別 shape。
- **case4 vs case8 = 2 file diff** (agent.py / physics.py) — lineage 内は近接。
- case8 core: geometry/safety **identical**, world_model 44行, physics 358行
  (t14 predict-cache=挙動等価), config 63行 delta。
- **runtime flag**: OM/lookahead **OFF** (実行されない)、HARASS/CRASH_EXPLOIT ON。
  → 実挙動 = plan_moves + 新 mission。case1 core_jax 再利用が効く。

### Step1 実装 (loop 原則: JAX vs 書き換え前 case8 Python)

- case1 core_jax 8 module を case8/baseline_jax/core_jax に **copy** (cross-case
  independence: import でなく複製)。
- config delta 適用: PARTIAL_SOURCE_MIN_SHIPS 6→16, REINFORCE_MAX_SOURCE_FRACTION
  0.75→1.0, REINFORCE_SAFETY_MARGIN 2→5, ROTATING_OPENING_VALUE_MULT 0.9→0.95,
  DENSE_ROTATING_NEUTRAL_SCORE_MULT 0.86→0.90。
- 結合テスト `tests/e2e/pipeline/rulebase/case8/test_agent_jax_identity.py` 作成
  (smoke + 10-game tripwire ≥3 gate、case1 と同方法論)。

### 検証

- import clean (compute_actions_jax_jit OK)。
- **smoke test 2 passed** (2×500turn、NaN/shape 異常なし)。~170s/game (case8 Python が
  predict-cache+多 mission で case1 14s より重い)、≤10min/game 制約内。
- tripwire (10-game ≥3) は実行中 → 結果は経過2 で記録。

### 残

- tripwire 結果確認。劣化 (≈0勝) なら config/mission delta を bottom-up に詰める。
- 新 mission (harass/crash_exploit) の JAX parity は未実装 (case1 core_jax は OFF 相当)。
  tripwire pass なら段階的に追加、fail なら原因切り分け。

## 経過 2 — tripwire 実行中 (loop 早期再 fire)、harass 分析で次段準備

tripwire (10-game) は ~28min 要し、loop が早期再 fire (1:25 経過時点で実行中)。
CPU 競合を避け重い新規実行はせず、次段の準備 (mission JAX 化の feasibility) を分析。

### harass mission 分析 (port feasibility)

case8 missions/harass.py は enemy × source の pairwise scan:
`plan_shot(src,target,probe)` → turns guard → `ships_needed_to_capture` → score
= stolen_production / (need + turns·w + 1)。**使う部品 (plan_shot,
ships_needed_to_capture) は case1 core_jax で実装済**。→ harass は case1 capture
と同型の vmapped pair scan として JAX 化可能 (feasible)。crash_exploit も同様に
要確認だが pairwise scan が基本構造。

→ tripwire pass なら、新 mission を core_jax に vmapped pair で追加し parity test を
bottom-up に。fail なら config delta の不足を切り分け。次 tick で tripwire 結果確認。

## 経過 3 — config delta 完全性を検証 (tripwire 継続中)

tripwire (pid 16445) は 2:27 経過時点で継続中 (10-game × ~170s/game ≈ 28min 見込)。
CPU 競合 + parity test の x64 競合を避け、純解析のみ実施。

### 検証: 現 port の config delta は完全

jax core が参照する 91 定数を抽出し case1↔case8 baseline config を全比較
→ **既に patch 済の 5 定数以外に未反映 delta なし**。残る case8 delta (HARASS_*,
CRASH_*, SWARM_MIN_PARTICIPANT_SHIPS, FULL_COMMIT_*) は **未 port の mission 専用**で、
現 capture/reinforce pipeline には影響しない。→ 現 Step1 port は config 的に完全。

### 判断: harass 実装は tripwire 結果待ち

harass の JAX 実装は feasible (経過2) だが、**未検証の base の上に積むのは loop の
「未検証実装回避」原則に反する**。現 base が非劣化と確認できてから mission を段階追加する。
次 tick で tripwire 結果 → pass なら harass/crash を bottom-up parity 付きで追加。
