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

## 限界 (honest)

case7 の t14 trap (STAY が ships 一斉発射で自滅) と異なり、case6 では **STAY が Python を
有利化**。stateless JAX では STAY (跨ターン状態) を faithful に持てず、33% に留まる。
非劣化 (≈0勝回避) は満たすが、case6 は本 lineage で最も parity gap が大きい case。
stateful STAY の近似 (e.g. EnvState から consecutive_holds を推定) は別途要検討。

→ **case6 完了** (config-only port + 非劣化 33%、STAY gap documented)。
