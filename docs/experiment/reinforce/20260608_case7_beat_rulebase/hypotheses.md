> 作成日: 2026-06-08 / 最終更新: 2026-06-08 / 状態: in_progress
> 最大 iteration: 制限なし(短時間PoCを反復) / 主要メトリクス: vs rulebase 勝率(paired) + 学習曲線 trend
> 既定 episode 数: paired 30-60戦(採否), 学習中 in-JAX eval は ep8

# case7 PFSP RL — ルールベースモデルへの勝利を目指す実験ループ

## 目標
JAX化ルールベース(baseline_jax_full / baseline_v8=case8)に **勝率 >50%** で勝てる
RL agent を、短時間GPU PoC(~20-30分/run)の反復で実現する。

## 現状(2026-06-08 baseline)
- fast_probe 20iter で best_win 0.812(vs noop)だが、**vs baseline_v8 = 0/10**。
- self_snapshot は ~0.5(互角)、baseline_jax_full は ~0.22。強相手未攻略。
- memory: 強すぎ相手直接学習は勾配破壊(`unbeatable_opponent_harmful`)、小規模は本物相手0勝が天井(`case6_live_eval`)。

## 実施しない検証 / 評価 (skip list)
### 評価
- Kaggle publicScore は引用しない (memory `project_om_finding`, `project_case5_validation`)
- 学習中の採否は **paired-seed 30-60戦** + 学習曲線 trend で行う(300戦は最終候補のみ)
### 分析
- n<300 結果で結論を出さない(採否は paired で分散低減、最終確認は n≥300)
- replay 詳細分析は当面しない(勝率 trend 優先)
### 実行
- Kaggle submit / promote はしない(本ループの対象外、別途承認要)
- 各 run 後は pod を destroy(課金停止)
### 例外条件
- ある施策が paired で +有意なら n=300 + 別 rulebase で最終確認

## 仮説リスト (priority 順)
- [x] (P1) H1: handicap curriculum — REJECTED (iter1)。lite ですら勝てず entropy collapse、難度調整は無効。ボトルネックは reward 信号
- [x] (P1) H2: dense差分報酬 — REJECTED (iter2)。max 0.375、H1と同じ振動+entropy collapse。reward shaping も plateau 破れず
- [x] (P1) H3: BC warm-start — REJECTED (iter3, harmful)。case9 imitation が case7 に転移せず vs noop 0.22/vs full 0.03。kl anchor が弱 policy に固定
- [x] (P1) H4: **scale-up 150iter — BREAKTHROUGH** (iter4)。vs full 0.23→0.34→0.47 単調上昇、勝ち越し(0.50-0.62)到達。天井は under-training だった
- [ ] (P1) H5: **scale-up 300iter** — H4 の climb を継続、勝ち越しを定着させる。最終 ckpt を paired 30戦で確定 (進行中)
- [ ] (P3) H6: reverse curriculum — H4/H5 で勝てれば不要。勝ち足りなければ補助 (research 処方A)
- [ ] (P3) H6: asymmetric reward — 弱側(agent)の勝ち報酬増幅/負け減衰で初期正信号確保 (research 処方E)

## 知見 (2026-06-08, H1/H2/H3 後 — round 1 結論)
- **3つの 20-iter 機構改変 (H1 opponent / H2 reward / H3 BC init) が全て rulebase を破れず**:
  - H1 handicap: vs full ~0.32 (trend無)
  - H2 dense差分: vs full mean 0.260/max 0.375
  - H3 BC warm-start: vs full 0.03-0.06 (**逆効果**、case9→case7 転移失敗)
- 共通: vs baseline_jax_full で **~0.3 が天井**、entropy collapse、学習 trend なし。
- **結論**: 20-iter 規模では機構をどういじっても rulebase に勝てない (memory `case6_live_eval`
  の「小規模 0/30 天井」を再確認)。ボトルネックは機構でなく **規模(iter数)**。
- → **次ラウンドは scale-up (H4)**: 最も素直な設定 (self_snapshot pool + dense差分, BC無し,
  kl_beta=0) で iterations を 150-200 に増やし、20iter ceiling が単なる under-training か
  本質的天井かを判定する。これが分岐点。
- 副産物の改善点: launch_poc.sh に BC scp 同梱、family=reinforce 明示で onstart 誤分類回避。

## 評価プロトコル(各 PoC 共通)
1. fast_probe_gpu ベースで施策を1つ変更 → RunPod 4090 で ~20分学習(中間 ckpt+metrics は S3)。
2. 学習中の in-JAX eval(opponent別 win)+ 完走後 metrics の trend を見る。
3. 最終 ckpt を **baseline_v8 と paired 30戦**(同一seed)+ baseline_v4 でも確認。
4. baseline(前 best)との Δ を paired で比較。+有意なら採用、ckpt を次ループの warm-start に。
5. pod destroy。結果を iter{N}_result.md に記録、hypotheses のチェックボックス更新。

## Iteration log
| iter | 開始 | 仮説# | run_id | 主要メトリクス | 採否 | result |
|---|---|---|---|---|---|---|
| (baseline) | 2026-06-06 | — | ...661d5ad | vs v8 0/10, self ~0.5 | — | docs/plans/case7-pool-rl/10 |
| 1 | 2026-06-08 | H1 | ...90444c5 | vs full ~0.32 (trend無), lite ~0.22 | REJECTED | iter1_result.md |
| 2 | 2026-06-08 | H2 | ...a1c5e8b | vs full mean 0.260/max 0.375 (trend無) | REJECTED | iter2_result.md |
| 3 | 2026-06-08 | H3 | ...acbab5c | vs full 0.03-0.06 (BC init 逆効果) | REJECTED | iter3_result.md |
| 4 | 2026-06-08 | H4 | ...500bc5d | vs full 0.23→0.47 climb, peak 0.62 | **BREAKTHROUGH** | iter4_result.md |
| 5 | 2026-06-08 | H5 | ...500bc5d(再利用) | 300iter, 進行中 | running | — |
