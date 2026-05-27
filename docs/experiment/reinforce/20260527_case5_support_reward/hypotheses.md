# Hypotheses — reinforce/case5 support_reward

> 作成日: 2026-05-27
> 最終更新: 2026-05-28 (iter2 analysis 完了、H2 ratio adopted)
> 状態: in_progress
> 最大 iteration: リスト消化まで (deepen も許可)
> 主要メトリクス: 合成指標 — vs `baseline_jax_lite` の **last-10 iter 平均 win_rate** と 学習 **reward trend (右肩上がり度)** を同時評価 (収束速度 + 最終性能)
> 既定 episode 数: 128 / iter (case3 確立レシピを継承した `kaggle_jax_train.yaml` 準拠、sample variance 半減済)

## スコープと固定軸

case5 の **support (補助) reward 追加** にスコープを絞る。現行報酬は
`terminal ±1 + shaping_coef · Δ(mine − enemy)` で、`shaping_mode` は
`ships` / `planets` の **排他**二択 (採用値 `planets`, `coef=0.50`)。本 case では
ship / 惑星の **保持割合・保持数** 等を補助報酬として加え、収束速度と最終性能の
向上を検証する。

以下は **固定** (case2 ablation で確立済みの学習レシピ、`kaggle_jax_train.yaml`)。
report shaping 以外の変更は本 case では行わない。

| 軸 | 固定値 | 出典 |
|----|--------|------|
| Backbone / Head | case1 純正 PerPlanetHead (from_head 無し) | case5 = case3 のレシピ継承 (case3 自体は case2 から from_head 除去) |
| opponent | curriculum (early=noop / late=baseline_jax_lite, switch_iter=5) | H4 best |
| lr / decay | 3e-5 → 3e-6 線形 (lr_schedule_steps=100000) | H + D |
| gamma / gae_lambda | 0.995 / 0.95 | — |
| entropy_coef / target_kl | 0.02 / 0.02 | G |
| episodes_per_iter / horizon | 128 / 500 | S / J 撤回 |
| iterations | 200 (long-run) | H6 |
| 既存 shaping baseline | `shaping_mode=planets`, `shaping_coef=0.50` | Y / F |

## 実施しない検証 / 評価 (skip list)

### 評価
- ローカル self-play 300 対戦は実施しない (学習中の last-10 win_rate (vs baseline_jax_lite, in-training) と reward trend のみで採否)
- Kaggle publicScore は引用しない (project rule)
- skill rating は使わない (project rule)

### 分析
- n<300 結果で結論を出さない (default ON、memory `project_imitation_case1_phase3` 由来)
- replay 分析 (experiment-analysis) は実施する (300 対戦 skip のため、采否は学習メトリクス主体だが replay は補助として残す)

### 実行
- なし (smoke test / `dev/test-bot` / RunPod GPU / auto-recover はすべて実施)
- case5 JAX 学習は 24GB+ VRAM 必須 (memory `project_runpod_a4000_oom`)。A4000 16GB は避け RTX 3090/4090 系を使う

### 例外条件
- H3 (絶対保持数の非差分加算) は potential-based でなく policy をバイアスさせる可能性 (Ng 1999)。
  inconclusive ではなく **明確に劣化/引き伸ばし傾向が出た場合** はそこで rejected とし deepen しない。

## 仮説リスト (priority 順)

- [x] (P1) H1: ship 差分と planet 差分を **同時併用** shaping (`coef_ship·Δship + coef_planet·Δplanet`) — 現状は排他二択。両領域の状態を同時に密フィードバック。potential-based を保ちバイアス無し。 — **inconclusive (trend は adopted 寄り)** (iter1: lite phase last-10 0.549 / trend +0.376, baseline 比 +~5pp。max approx_kl 0.0055 で安定、200 iter 完走。n<300 で確定保留)
- [x] (P1) H2: 保持「**割合**」差分 shaping (potential = `mine/(mine+enemy)` を ship・planet で算出し、その turn 差分を報酬) — 絶対数の production スケール依存を排し [0,1] 正規化で係数調整が容易。 — **adopted** (iter2: lite last-10 **0.763** / trend +0.651, H1 比 **+21pp**。割合正規化で value_loss 0.43→0.005 が決定打。max approx_kl 0.0047 で安定。n<300 で確定保留だが noise floor 大幅超過)
- [ ] (P2, depends on H1) H4: 併用時の `coef_ship : coef_planet` **比率 sweep** (例 0.5:0.5 / 0.25:0.75 / 0.75:0.25) — planets=0.50 単体が現 best なので ship 成分の最適追加量を探る。H1 採用が前提。
- [ ] (P2) H5: **production potential** 補助 (保有惑星の production 合計の差分を shaping) — 単なる惑星数より高 production 惑星の保持を評価。territorial 信号の質向上。
- [ ] (P2, depends on H2) H7: 保持割合差分の **clip / 正規化** で報酬スケール安定化 (H2 派生) — 序盤の割合急変による spike を抑え value_loss を安定化。H2 採用が前提。
- [ ] (P3, 対照) H3: **絶対保持数の非差分 dense 加算** (`coef · (mine_ships or mine_planets)` を毎 turn) — 生存・拡張の直接報酬。⚠️ 非 potential-based で最適方策をバイアス (貯め込み/引き伸ばしリスク)。potential-based 系との性能差を見る対照群。
- [ ] (P3, pair with H3) H6: 勝ちターン短縮 **time bonus** (早期勝利に terminal bonus / 引き伸ばしに小ペナルティ) — dense 保持報酬の「引き伸ばし」副作用への対策。H3 と pair で効果を見る。

## Iteration log

(各 iter 完了時に experiment-analysis / experiment が追記)

| iter | 開始 | 仮説# | plan path | run_id | 主要メトリクス | 採否 | result path |
|---|---|---|---|---|---|---|---|
| 1 | 2026-05-27T15:16Z | H1 | iter1_plan.md | 20260527-151636__feature-support-reward__2f37b9e__seed0 | last-10 0.549 / trend +0.376 (lite phase) | inconclusive (trend は adopted 寄り) | iter1_result.md (analysis: 学習ログ baseで実施、replay skip) |
| 2 | 2026-05-27T18:23Z | H2 | iter2_plan.md | 20260527-182312__feature-support-reward__c359b68__seed0 | last-10 0.763 / trend +0.651 (lite phase, H1 比 +21pp) | adopted | iter2_result.md (analysis: 学習ログ baseで実施、replay skip) |

## 参考 (References)

- [Ng, Harada, Russell (1999) — Policy Invariance under Reward Transformations](https://www.emergentmind.com/topics/potential-based-reward-shaping) — PBRS は potential 関数 Φ の状態間差分で報酬を整形すれば最適方策が不変。現 case5 の Δ(mine−enemy) shaping はこの族に属し、H1/H2/H5/H7 も potential-based で設計。H3 の絶対量加算は非 PBRS でバイアス源。
- [Improving the Effectiveness of Potential-Based Reward Shaping in RL (arXiv 2502.01307)](https://arxiv.org/html/2502.01307v1) — PBRS の効果は初期 Q 値・外部報酬とのバランス依存。shaping_coef のスケール調整 (H4/H7) が sample 効率に効く根拠。
- [Reward Shaping for Improved Learning in RTS Game Play (arXiv 2311.16339)](https://arxiv.org/pdf/2311.16339) — RTS でゲームイベント別の shaping が勝率と学習時間を改善し得る一方、不適切な設計は害。ship/planet/economy など複数信号の併用 (H1/H5) と、害になり得る項 (H3) の対照という構成を支持。
