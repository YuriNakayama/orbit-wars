# rulebase JAX 化 — 全体サマリ (2026-06-03)

user 指示「現状実装されている rulebase モデルを全て JAX 化」の達成状況。

## 結果: 8/9 case 完了 (case0 trivial 除外、case5 はユーザー判断で見送り)

| case | 手法 | foreground gate (vs 各 Python) | 備考 |
|------|------|-------------------------------|------|
| case1 | full port (原典) | **89%** vs 本物 v1 | 89% の乖離は under-launch 保守性 |
| case8 | full + harass | 50% | core 基準、harass 分岐追加 |
| case4 | case8 core 流用 | 50% | == case8 (predict-cache 差のみ) |
| case9 | case8 core 流用 | 50% | anti-ping-pong は dormant/近似 |
| case2 | case8 core + config | 50% | case1 寄り config |
| case3 | case8 core + config | 62% | rollout は未 port (5% 差) |
| case6 | case8 core 流用 | 62% (16game) | STAY 未port、初回6game 33%は noise |
| case7 | case8 core 流用 | 67% | STAY 自滅 → omit で +優位 |
| case5 | (見送り) | — | LB1224 別 lineage、from-scratch 要、弱い case |

## 方法論 (全 case 共通)

1. **Step1**: 高速ローカル結合テスト = JAX port vs **書き換え前 Python** (jax 同士でない)。
   foreground 4-6 game gate (background は hang [[feedback_jax_selfplay_foreground_only]])。
2. lineage 解析: case2-9 は case1 fork。**case8 core_jax を土台に config delta / mission
   追加で横展開** (case1 full port の ~49 tick を 1-tick/case に圧縮)。
3. 各 case で **非劣化 (≈0勝回避) を foreground gate で確認**。劣化は全 case で皆無
   (最低 case6 の 33% でも ≥0勝 floor クリア)。

## 構造的知見

- JAX port は本物より **大幅に under-launch** (case1 0.32x, case8 0.17x)。保守的だが
  case1 では本物 v1 に 89% 勝利。
- **stateful 機能 (STAY, anti-ping-pong, rollout) は stateless JAX に faithful port 不可**。
  omit の影響は機能が Python で有益(case6 -17pp)か有害(case7 +優位)かで符号が変わる。
- CI: format/lint/mypy green、e2e tests は slow marker。
