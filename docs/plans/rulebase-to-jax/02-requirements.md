# rulebase-to-jax — Requirements Definition

## Background and Purpose

現状 `bot/pipeline/rulebase/case0–9` の rulebase agent は純 Python 実装で、Kaggle submit 用には十分だが **RL 訓練の self-play opponent としては遅い**。reinforce/case6 PFSP は JAX-native rollout を持つが、opponent pool は case1 の lite/full の 2 種のみで多様性が乏しく、memory `project_reinforce_case6_live_eval` が示す **train(JAX近似rule)/eval(本物v1) ギャップ**が RL 進歩の live 転移を阻んでいる。

champion case4・LB1224 case5・baseline case1・harass/swarm case2 を **full parity で JAX port** し、vmapped self-play の opponent pool に多様性と忠実度を与えることが目的。

> ⚠️ **非目的**: Kaggle submit agent の JAX 置換は行わない。bench 上 per-turn 5.6× 遅く、1秒/turn 予算で timeout リスク。JAX port は **offline GPU 専用**。

## User Stories

- RL 研究者として、case1/2/4/5 の忠実な JAX opponent を使い、vmapped self-play を高速に回したい。多様な強敵で訓練し v1 への live 勝率を上げるため。
- RL 研究者として、opponent を strategy_id で切替えられる共通インターフェースで vmap したい。PFSP pool に分岐レスで組み込むため。
- 開発者として、各 JAX port が Python 版と action 完全一致することを parity test で保証したい。train/eval ギャップを最小化するため。

## Functional Requirements

1. `case1/2/4/5` 各々に `baseline_jax/` (full parity 版) を実装。入力は `orbit_wars_jax.EnvState`、出力は `(MAX_LAUNCHES_PER_AGENT, 3)` action tensor。
2. 5 mission (snipe/reinforce/harass/swarm/evacuate) + fleet consolidation を **全 mission 並列 score → mask → argmax** で表現（`lax.switch` 不使用）。
3. world_model の 8-turn 防衛シミュレーションを `lax.scan(length=HORIZON)` + 固定 shape carry + active mask で port。
4. ray-circle fleet-target 判定を判別式 + `jnp.where` の分岐レス閉形式で port。
5. 全 agent を **strategy_id を data で受ける分岐レス共通インターフェース**に統一し、1 関数 vmap で異種 opponent を回せるようにする。
6. 各 port に対し Python 版との **action 一致率 parity test** (tie-break 統一, TDD で先に書く) を整備。
7. reinforce/case6 rollout の opponent enum に新 port を登録（`OPPONENT_*`）。

## Non-Functional Requirements

- **Parity (最重要)**: 同一 obs 大量サンプルで Python 版と action 一致率を測定。tie-break (index 最小等) を両実装で統一。整数 ID/index/mask は完全一致、float は rtol 1e-5。
- **GPU throughput**: RunPod GPU で vmapped self-play の episode/s を計測し rust/python backend と比較。`_bench/` パターン踏襲。
- **Live 勝率**: 各 port を本物 Python 版と 300 戦させ互角(parity 健全性の最終関所)を確認。
- jit/vmap friendly: 固定 shape + mask、Python int seed の jit 再 trace 回避。

## Out of Scope

- Kaggle submit agent の JAX 置換（逆効果のため明示的に除外）。
- case0(archive) / case3 / case6–9 の port（case4 派生で重複、本イテレーション外）。
- RL 訓練への新 opponent 組み込み後の勝率改善実験そのもの（別 /experiment ループの責務）。
- adapter (`orbit_wars_sim`) への JAX env 登録（既存通り直 import）。

## Glossary

- **full parity**: 同一 obs で Python 版と action が完全一致するレベルの port 忠実度。
- **strategy_id**: 共通 opponent 関数に data として渡す、どの case の戦略を実行するかの識別子。
- **PFSP**: Prioritized Fictitious Self-Play。case6 の opponent pool 方式。
