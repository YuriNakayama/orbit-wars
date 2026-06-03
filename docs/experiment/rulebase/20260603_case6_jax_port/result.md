# rulebase/case6 JAX port — result

> 記録: 2026-06-03 ~14:30 / 状態: done (Step1+非劣化、但し STAY gap) / 親: ../20260603_case8_jax_port/

## case6 = case8-lineage + STAY judge (ACTIVE, stateful)

- 差別化 = **STAY judge** (source に ships を hold、ACCUMULATE/burst)。STAY_ENABLED=True。
- **active**: case6 Python vs case8 Python 乖離 seed0=0% だが **seed1-3 で 48-76%**
  (first mismatch @turn1)。STAY は実戦で頻繁に発火。
- STAY は `consecutive_holds` を跨ターン保持 = **stateful、stateless JAX に port 不可**。
- jax-relevant config delta なし (capture/reinforce 定数は case8 と同一)。

## 実装 + 検証

- case8 core_jax を verbatim copy (STAY 未 port = config-only)。結合テスト作成。
- **foreground 6-game gate: JAX 2/6 = 33%**。≈0勝劣化の floor (≥3/10≈≥1.8/6) は超えるが
  50% を下回る。STAY が case6 Python を有利にしており、omit すると JAX が ~17pp 劣後。

## 追検証で 33% は noise と判明 (通算 62.5%)

初回 gate (seeds0-2) の 2/6=33% は n=6 の sample noise。disjoint seeds3-7 を追加実行:
**JAX 8/10 = 80%**。通算 **10/16 = 62.5%**。→ case6 JAX port は非劣化どころか case6
Python に勝ち越し。STAY omit の影響は懸念したほど大きくない (STAY は seed 依存で
有益/中立が振れる)。

## 限界 (honest)

STAY (跨ターン `consecutive_holds`) は **stateless JAX に faithful port 不可**につき
未 port。実戦では case6 Python に対し平均 62.5% で勝ち越すため degradation ではないが、
STAY が強く効く特定盤面では Python の挙動と乖離する。stateful STAY の近似は別途要検討。

→ **case6 完了** (config-only port + 非劣化、通算 62.5%)。教訓: gate は n≥10 で判断
(n=6 では 33%↔80% に振れる)。
