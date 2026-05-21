# Hypotheses — reinforce/case1 ppo_stabilization

> 作成日: 2026-05-18
> 最終更新: 2026-05-18 (iter1 analysis 完了)
> 状態: in_progress
> 最大 iteration: 10 (deepen も許可)
> 主要メトリクス: 安定化合成指標 — (a) `max approx_kl` が全 iter で < 0.1、(b) `win_rate vs baseline_v1` の iter 間 std が単調 ↓、を同時に評価
> 既定 episode 数: 32 ep / iter

## スコープと固定軸

case1 は **PPO 学習の安定化** にスコープを絞る。以下は **固定** (head 変更による表現力低下は禁止というユーザー指示)。

| 軸 | 固定値 | 出典 |
|----|--------|------|
| Backbone | Set Transformer (ISAB×4, hidden=192, attn=8, m=24) | case9 per_planet と同型 (BC 重みを strict=False で 1:1 ロード) |
| Head | `head_per_planet` (per-source × (P+1) pointer attention) + `value_head` (新規) + `ship_log_std` (新規 nn.Parameter) | `bot/pipeline/reinforce/case1/policy/heads/` |
| Action 分布 | Categorical(P+1) per src + Normal(log1p ships) | factorized log_prob で sum |
| BC warm-start | `data/output/models/imitation/case9_per_planet/runs/20260512-080505__.../best.pt` | DVC 管理、strict=False |
| Featurizer / Backbone weights | 不変 | head dim 削減 / dropout 強化 / head 簡素化系の仮説は **不採用** |

可変軸: PPO ハイパーパラメータ + 安定化テクニックのみ。

## 実施しない検証 / 評価 (skip list)

### 評価
- **300 対戦評価をしない** — `eval_vs_baseline` は 50 ep までで止める。安定化検証は loss / approx_kl / bc_kl 推移を主軸とし、n<300 win_rate は **inconclusive 固定**
- Kaggle publicScore は引用しない (project rule、memories `project_om_finding` / `project_case5_validation`)
- skill rating は使わない (project rule)

### 分析
- **replay 分析 (`experiment-analysis` の replay 解釈) は実施しない** — 学習曲線で採否判定
- **n<300 結果で結論を出さない** (memory `project_imitation_case1_phase3_non_determinism`) — 採否は (a) max approx_kl、(b) policy_loss / value_loss / bc_kl curve の単調性、で判定

### 実行
- **ローカル CPU を一切使わない** — local CPU 学習 / rollout collection は全面禁止 (リソース逼迫のため)。1-episode smoke test も pod 上で実行する
- dev/test-bot は RunPod 投入前に実施
- RunPod auto-recover は使用

### 例外条件
- ある仮説で (a) と (b) の両方が大幅改善 (max approx_kl < 0.05 + bc_kl monotonic decrease) を達成した場合、その weight に対してのみ 100 ep eval を追加実行可能 (ユーザー判断、本 hypotheses.md のスコープ外)

## 仮説リスト (priority 順)

- [x] (P1) **H1: target_kl early stopping** — `ppo_update` に `target_kl=0.05` を追加し、各 epoch の minibatch loop 内で平均 approx_kl が target_kl を超えたら当該 epoch を即 break。PPO 標準実装 (CleanRL / SB3) で、clipping 単独では守れない trust region を hard 保証。**期待効果**: 高 approx_kl が観測された後に更にダメージを拡大しない、bc_kl の monotonic 低下を回復 — **inconclusive (iter1)**: 機構は機能 (epochs_run mean=1.43/2、57/100 iter で early stop)、bc_kl std=0.14 で安定化方向。しかし max approx_kl=1.57 で primary metric (a) 未達成。判定が epoch 単位 minibatch 平均粒度のため、epoch 内単発 spike を抑止できない。target_kl=0.05 設定は副作用なしで継続。

- [ ] (P1) **H2: lr 1.0e-4 → 3.0e-5 (低い lr) **— `training.lr` を 3.3× 下げる。per-update の weight 変動を小さくし、approx_kl の山を平坦化。配布値 (Schulman 2017 や CleanRL の MuJoCo recipe) を下回る attempt。**期待効果**: max approx_kl の急峻なピークが消える代わりに学習速度が遅くなる

- [ ] (P1) **H3: kl_beta 0.5 → 0.1** — BC anchor 拘束を弱める。web 調査 (KL-regularized fine-tuning literature, 2025) で **β=0.001-0.1 が標準範囲**、現状 0.5 は **過剰拘束**で BC 重みからほぼ動けず advantage signal を消費しきれていない可能性。**期待効果**: bc_kl は若干上昇するが win_rate も上昇

- [ ] (P2) **H4: lr scheduler (linear decay)** — `optim.lr_scheduler.LinearLR` で iter 経過とともに lr を 1.0× → 0.1× へ線形 decay。後半 iter の trust region をさらに自動的に狭める。CleanRL の MuJoCo recipe で標準。**期待効果**: iter 後半の approx_kl が抑制され bc_kl curve が安定

- [ ] (P2) **H5: episodes_per_iter 16 → 32** — advantage 推定の variance を低減。1 iter のサンプル数 16 ep × ~500 turn ≈ 8k → 16k transitions に倍増。**期待効果**: advantage normalization (現状 per-batch z-score) の母集団が大きくなり policy ratio が極端値を取りにくくなる。**RunPod コスト 2×**

- [ ] (P2) **H6: value loss clipping** — `value_loss = max((v_new - v_target)^2, (clipped_v - v_target)^2)` で value head の per-update 変化幅を `clip_eps` (=0.2) 内に制限。CleanRL/SB3 標準実装。**期待効果**: value head 過更新による policy gradient のノイズ低減 → approx_kl の急上昇を間接的に抑える

- [ ] (P3) **H7: opponent curriculum** — `opponent` を `random_noop` → `baseline_v1` に段階移行。具体的には iter 0-5 は random_noop、iter 6-15 は baseline_v1、それ以降は self-play snapshot (将来) などのスケジュール。**期待効果**: obs 分布シフトを緩やかにし、easy opponent で確実に勝てる policy ができてから難敵に移る

- [ ] (P3) **H8: no_op_bias を学習可能 nn.Parameter に** — 現状 `ModelConfig.no_op_bias=8.0` の定数。これを `nn.Parameter(torch.tensor(8.0))` にして policy が fire/no-op ratio を自分で調整可能に。**head 拡張 (param 1 個追加) だが表現力低下ではないので採用可** (ユーザー確認済み)。**期待効果**: BC の偏りを policy が漸進的に解消できる、初期の fire-spam を回避

## Iteration log

(各 iter 完了時に experiment-analysis / experiment が追記)

| iter | 開始 | 仮説# | plan path | run_id | 主要メトリクス | 採否 | result path | analysis path |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-05-18 | H1 (target_kl=0.05) | iter1_plan.md | 20260518-013025__feature-reinforce-learning-case0__f406f85__seed0 | max approx_kl=1.57 (>0.1: 27/100), bc_kl std=0.14, epochs_run mean=1.43/2 | inconclusive | iter1_result.md | iter1_analysis.md |

## 参考 (References)

- [Posterior Behavioral Cloning (PostBC, 2025)](https://arxiv.org/abs/2512.16911) — BC pretrain で action coverage が不足すると PPO finetune で KL 発散しやすい。`no_op_bias` は近似対策、温度補正が本質
- [KL-regularized fine-tuning overview](https://www.emergentmind.com/topics/kl-regularized-fine-tuning) — BC term β は `0.001 ≤ β ≤ 0.1` が標準。現状 β=0.5 は過剰拘束の可能性 (H3)
- [CleanRL PPO implementation](https://docs.cleanrl.dev/rl-algorithms/ppo/) — target_kl early stopping (H1) と value loss clipping (H6) のリファレンス
- [The 37 Implementation Details of PPO (ICLR 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — `approx_kl > target_kl` 時の epoch 中断は trust region を hard 保証する PPO 標準テクニック
- [PPOxFamily hybrid action tutorial](https://github.com/opendilab/PPOxFamily/blob/main/chapter2_action/hybrid_tutorial.py) — Categorical + Gaussian factorized PPO の log_prob は per-component sum で OK (現状実装と整合)
