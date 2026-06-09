# Hypotheses — reinforce/case8 pfsp_vmpo_vs_ppo

> 作成日: 2026-06-09
> 最終更新: 2026-06-09
> 状態: in_progress
> 最大 iteration: リスト消化まで
> 主要メトリクス: vs rulebase case8 (baseline_v8) held-out 勝率 (paired-seed) + 学習曲線 trend
> 既定 episode 数: 学習中 in-JAX eval は paired 30-60 戦、最終候補のみ ≥300

## 目標

reinforce で新 case (case8) を作成し、**V-MPO と PPO のどちらが有力か**を同一 harness 上で A/B する。

- **学習用対戦** = PFSP `f_var` で「自分と同実力(勝率 ~0.5)」の相手を選択。pool = `[case1 lite/full, rulebase case8, self-snapshot]`、self-snapshot は補助。
- **学習進捗評価** = 固定された rulebase **case8 (baseline_v8)** との戦績(held-out, 固定シード、PPO update なし)で測定 + Elo。
- **比較方式** = PPO loss は case7 `ppo_jax.py` を流用、**V-MPO loss を新規実装**して `algo` フラグで切替。

## 設計上の前提 (Phase 1 コード調査 — 2026-06-09)

case7 (`pipeline/reinforce/case7/`) が既に必要機構をほぼ保有しており、case8 はこれを土台にする:

- `training/ppo_jax.py`: clipped surrogate + value MSE + entropy + BC-KL anchor。`evaluate_actions_jax` / `make_optimizer` / GAE は V-MPO でも再利用可。**loss 関数のみ差し替え**れば V-MPO arm が組める。
- `training/train_jax.py`: `_PoolSelector`(`f_var`/`f_hard` の `_pfsp_weight`)、`_heldout_eval`(固定 opponent + 固定 `eval_seed`)、`_elo_update`(固定 ref anchor)。held-out opponent は YAML `heldout_eval.opponent` で指定可(現状 default `baseline_jax_full` → case8 では `baseline_v8` に変更)。
- `training/rollout_jax.py`: opponent_mode dispatch に `noop/lite/full/self_snapshot/python_v1/v4/v8` を実装済み(`lax.switch` lambda 化で host callback 律速回避済み)。**case8 = `python_v8` が pool・held-out の双方で利用可能**。
- config 例: `h5_fvar_heldout.yaml`(f_var + held-out)、`h6_fvar3pool.yaml`(3-opp pool)、`kaggle_jax_train_pool_v8.yaml`(v8 pool)が既にあり、case8 の出発点になる。

## 既知のリスク / 過去知見 (採否に反映)

- **from-scratch RL は competent 相手にブートストラップ不可(確定天井)** — case7 は PPO+PFSP+dense reward+BC warm-start を尽くしても vs baseline_v8 = 0/10、vs full ~0.22(memory `case7_train_eval_parity_gap`)。本実験の核は「rulebase に直接当てない」枠組み: train は self-play/同実力で 0.5 を保ち、case8 は **測るだけ**。V-MPO がこの天井を破れるかが本質的な問い。
- **勝てない強相手での直接学習は有害** — reward 飽和で勾配破壊(memory `reinforce_unbeatable_opponent_harmful`)。f_var で 0.5 付近に保つ設計はこれを回避する手段。
- **train/eval featurizer parity ギャップ** — in-JAX self-play の進歩が実環境に乗らない事例(memory `case6_live_eval`)。held-out は同 featurizer の case8 を使い、最終候補のみ別 instance / live で確認。
- **horizon は必ず 500**(memory `reinforce_horizon_terminal_reward_bug`)、**JAX self-play は foreground 4-game gate**(memory `jax_selfplay_foreground_only`)。
- **長時間 run は iter ごと S3 へ best.pt 即 upload**(規約 + memory `reinforce_case7_longrun_hang`)。

## 実施しない検証 / 評価 (skip list)

### 評価
- Kaggle publicScore は引用しない (project rule, memory `project_om_finding` / `project_case5_validation`)
- 学習中の採否は **paired-seed 30-60 戦** + 学習曲線 trend で行う(300 戦は最終候補のみ)
- skill rating は採否に使わない (project default)

### 分析
- n<300 結果で結論を出さない(採否は paired で分散低減、最終確認は n≥300)
- replay 詳細分析は実施しない(experiment-analysis は勝率集計のみの skip モードで回す)

### 実行
- Kaggle submit / promote はしない(本ループ対象外、別途承認要)
- 各 RunPod run 後は pod を destroy(課金停止)
- smoke test (1-episode self-play) は **必須**(skip しない)。JAX self-play hang 対策で foreground 4-game gate を使う

### 例外条件
- V-MPO が PPO に paired で +有意に勝った場合のみ、その設定を n=300 + 別 rulebase (case4/case1) で最終確認

## 仮説リスト (priority 順)

- [ ] (P1) H0: **case8 scaffold** — `pipeline/reinforce/case8/` を case7 コピーで作成し、`training/` に `algo: ppo|vmpo` フラグを通す。PFSP `f_var` / held-out / Elo の wiring を case8 向け(held-out=`baseline_v8`, pool=case1+case8+self-snapshot)に組み替え。smoke (1-ep) が両 algo で通ることが受入条件。— V-MPO/PPO を公平に A/B する同一 harness の土台
- [ ] (P1, depends on H0) H1: **V-MPO loss 新規実装** — top-half advantage で非パラメトリック target ψ(温度 η の dual L_η, ε_η 制約) + 重み付き最尤の policy loss L_π + decoupled trust-region L_α(Lagrange α, ε_α, old policy に stop-gradient)。importance weight / entropy reg なしで PPO と同一 rollout・同一 PFSP・同一 held-out 上で A/B。— PPO の clip/entropy 依存を外し、policy collapse 耐性で held-out case8 勝率が PPO を上回るかが主問い
- [ ] (P1, depends on H0) H2: **PFSP f_var pool** — 学習対戦相手 pool = `[case1 lite, case1 full, rulebase case8, self-snapshot FIFO]`、`f_var(x)=(x(1-x))^p` で勝率 0.5 付近の相手を優先選択。per-iter win_rate が ~0.5 に張り付くこと(設計通り)を確認。— 同実力対戦の非飽和維持が学習信号の前提(memory `unbeatable_opponent_harmful` 回避)
- [ ] (P1, depends on H0) H3: **held-out case8 + Elo** — 固定 `baseline_v8` + 固定 `eval_seed` で `heldout_eval_every` ごとに勝率測定(PPO update なし)、固定 ref anchor で Elo 更新。f_var 下では per-iter win_rate は進捗信号にならないので held-out 曲線/Elo を唯一の進捗軸にする。— PPO arm / V-MPO arm の進捗を同一スケールで比較する評価基盤
- [ ] (P2, depends on H1) H4: **V-MPO HP sweep** — V-MPO が PPO に勝つ兆候が出たら ε_η / ε_α / top-k 割合(default 0.5)を sweep し勝ち筋の感度を確認。— V-MPO 採用時の頑健性確認(深掘り)
- [ ] (P2, depends on H1) H5: **scale-up** — 勝った側 algo で iterations 20→150-200 にスケール(memory `case1_aa_300iter` は 300iter で reward 0.50 到達)。短 run の ceiling を破れるか、V-MPO の優位が長 run で太るか(賃 ~$1-1.5)。— 短 PoC の天井破りと優位の持続性検証

## Iteration log

(各 iter 完了時に experiment-analysis / experiment が追記)

| iter | 開始 | 仮説# | plan path | run_id | 主要メトリクス | 採否 | result path |
|---|---|---|---|---|---|---|---|

## 参考 (References)

- V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control (Song et al., DeepMind, ICLR 2020, arXiv:1909.12238) — 非パラメトリック target distribution を top-half advantage + 温度 η で構成 → 重み付き最尤で policy 更新 → decoupled KL trust-region(α）で制約。importance weighting / entropy reg / population-based tuning なしで Atari-57・DMLab-30 の従来 score を更新。policy collapse 耐性が PPO 比の主な売り。https://arxiv.org/abs/1909.12238 / https://openreview.net/forum?id=SylOlp4FvH
