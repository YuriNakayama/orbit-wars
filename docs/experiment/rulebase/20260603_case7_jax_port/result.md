# rulebase/case7 JAX port — result

> 記録: 2026-06-03 ~14:45 / 状態: done (Step1+優位) / 親: ../20260603_case8_jax_port/

## case7 = case8-lineage + STAY (t14 trap、自滅的)

- 差別化 = STAY (case6 と同系)。但し memory [[project_case7_t14_trap]]: case7 の
  stay/accumulate は **t14 で ~60 ships 一斉発射 → ~70% で大失血** = 自滅的 liability。
- jax-relevant config delta なし (case8 と同一定数)。

## 実装 + 検証 (仮説検証)

- case8 core_jax を verbatim copy (STAY 未 port)。結合テスト作成。
- **仮説**: 自滅的 STAY を omit すると JAX port は case7 Python より**強くなる**。
- **foreground 6-game gate: JAX 4/6 = 67%** → **仮説確認**。case6 (STAY が有利 → 33%) と
  対照的に、case7 (STAY が liability → 67%)。memory の t14 trap を裏付け。

→ **case7 完了** (config-only port、STAY omit が優位に作用、非劣化どころか勝ち越し)。

## lineage まとめ (STAY/rollout omit の影響は case 依存)

| case | 差別化機能 | gate | omit の影響 |
|------|-----------|------|------------|
| case2/4/9 | (dormant/config) | 50% | 中立 |
| case3 | rollout | 62% | ほぼ中立 (5%差) |
| case6 | STAY (有益) | 33% | -17pp |
| case7 | STAY (自滅) | 67% | +優位 |

stateful 機能 (STAY) を stateless JAX で省略した影響は、その機能が Python で有益か
有害かで符号が変わる。全 case で ≈0勝劣化は回避。
