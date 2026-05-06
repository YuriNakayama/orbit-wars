# Rulebase Cases

Pure-Python rule-based agents. Each case is an independent submission package
(see `.claude/rules/pipeline.md`).

case1 (sigmaborov LB897 port) を起点に、mission / movement / lookahead / fleet 制御を
段階的に積み上げる系譜。case4 が production champion (LB745)、以降は派生実験。

## Status table

| Case | レジストリ名 | Status | 戦略要点 | publicScore | 備考 |
|------|-------------|--------|---------|-------------|------|
| case0 | `case0` | archive | 単純スナイパー参考実装 | n/a | 学習用、refactor 対象外 |
| case1 | `baseline_v1` | active (legacy) | sigmaborov LB897 port (sniper + reinforcement)。`strategy.py` を `planner/` に分割 | LB 897 | 全 case の出発点 |
| case2 | `baseline_v2` | active | case1 + opponent model + lookahead + harass / swarm mission + evacuation/followup/rear_guard movements | n/a | OM ablation 結果あり (デフォルト OFF) |
| case3 | `baseline_v3` | active | case2 + 内蔵ロールアウト (`lookahead/rollout.py` 325 行)。mission scoring に短期 rollout 結果を反映 | n/a | Phase A/B/C で V/DENSE 採用 (300 戦 70.3%) |
| case4 | `baseline_v4` | **production** | case3 + `missions/fleet_consolidation.py` (余剰艦の集約による効率向上) | 745 | 現役チャンピオン。turn_p95 が actTimeout 上限張り付き |
| case5 | `baseline_v5` | active (verification) | Roman Tamrazov LB1224 notebook の verbatim port (`agent_full.py` 2455 行) | 600 | 自己対戦 vs case4 56% / vs case1 70% も Kaggle publicScore は case4 を下回る |
| case6 | `baseline_v6` | active (experiment) | case4 のフルコピー + **STAY 判定** (発射せず保留する STAY_BURST 1 ターン arbitrage) | n/a | defense hold + burst hold |
| case7 | `baseline_v7` | active (experiment) | case6 + **多ターン蓄積からの遠距離単発攻撃 mission** (ACCUMULATE_BURST) | n/a | t14 ship-loss trap が確認済み (60 ships 一斉発射) |
| case8 | `baseline_v8` | **採用版** | case4 base + `physics.predict_planet_position` の **dict cache 化**。挙動完全等価で速度最適化 | n/a | 200 戦 50.5%、turn_p95 -25%。multi-step 最適化試行 (beam/PGS/NaïveMCTS 等) の集約先 |
| case9 | `baseline_v9` | rejected (anti-ping-pong) | case4 + cooldown bypass + plan_shot cache | n/a | 9 iter 探索で +5pp 不可確定 (真値 ~50%)。iter6 plan_shot cache のみ採用、`docs/experiment/rulebase/20260504_case9_anti_ping_pong/` |

## 系譜

```
case0  archive (sniper 参考実装)

case1 (sigmaborov LB897 port)
  └─ case2 (+ OM, lookahead, harass, swarm, movements)
       └─ case3 (+ rollout)
            └─ case4 (+ fleet_consolidation)  ★ production LB745
                 ├─ case6 (+ STAY judge)
                 │    └─ case7 (+ multi-turn accumulate burst)
                 ├─ case8 (predict cache, 速度最適化、採用版)
                 └─ case9 (anti-ping-pong, rejected)

case5  独立 (Roman Tamrazov LB1224 notebook の verbatim port)
```

## Conventions

- `case<N>/baseline/` が agent body
- `evaluation/snapshot_update.py` は `src/evaluation/snapshot_update.py` 経由
- 大型 `strategy.py` は `case<N>/baseline/planner/` に分割可 (case1 が参考実装)
- cross-case import は禁止 (case 内で完結。共通化したい場合は重複保持)

## モデルバージョン命名規則 (case 内バージョニング)

rulebase は学習を伴わないため重みファイルを持たない。戦略パラメータ tuning の
iter 管理は `docs/experiment/rulebase/{yyyymmdd}_case<N>_{topic}/iter<N>_{plan,result}.md`
で行う。case 番号は時系列に連番。

詳細: `docs/plans/refactor-directory/`
