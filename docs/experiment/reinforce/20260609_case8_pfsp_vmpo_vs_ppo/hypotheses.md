# Hypotheses — reinforce/case8 pfsp_vmpo_vs_ppo

> 作成日: 2026-06-09
> 最終更新: 2026-06-09
> 状態: in_progress
> 最大 iteration: リスト消化まで
> 主要メトリクス: ① pool 内 per-iter 勝率(目標 ~0.5)② 固定相手 held-out 勝率(0 から滑らかに増加)③ 速度(20 iter ≤ 30分 on GPU)
> 既定 episode 数: in-JAX eval は paired 32戦、最終候補のみ本物 case8 と offline paired 300戦

## 目標

reinforce/case8 で **PPO と V-MPO のどちら(の振る舞い)が良いか**を厳密な A/B で検証する。

**実験規律(最重要)**: A/B テストは **1 パラメータのみ変更**して比較する。
- **Phase 1 で PPO の適切な実験条件(固定相手・pool 構成・iteration 数・LR 等)を確定・凍結**する。
- **Phase 2/3 は Phase 1 で凍結した条件をそのまま使い、`algo`(と Phase 3 では V-MPO 内部 HP)だけを変える**。
- 実験条件を Phase 2/3 で変えると、どんな結果も PPO と比較不能になり無意味化する。これを絶対に避ける。

全実験は **JAX で GPU 上**実行。本物 rulebase (python_v8) は host-callback のため学習ループに入れない
(in-loop は in-JAX proxy、本物 case8 比較は学習後 offline)。

## 設計上の前提 (コード調査 + iter1 実測)

- `training/ppo_jax.py`: clipped surrogate + value MSE + entropy + BC-KL。`evaluate_actions_jax` /
  `make_optimizer` / GAE は algo 非依存で V-MPO でも再利用。**loss 関数のみ差し替え**で V-MPO arm が組める。
- `training/train_jax.py`: `algo: ppo|vmpo` フラグ配線済(H0)。`_PoolSelector`(f_var)/`_heldout_eval`
  (固定 opponent + 固定 seed)/`_elo_update` 実装済。
- **iter1 実測(H0)**: GPU RTX4090 で定常 ~23s/iter(iter0 のみ compile 61s)。in-JAX opponent
  (noop/lite/full/self)のみなら GPU 稼働。held-out=`baseline_jax_full`(in-JAX)で進捗測定可。
- iter1 観察: noop 区間 win 0.72-0.88、pool 切替(iter4)で 0.25 + entropy 48→14 急落。
  **f_var が 0.5 に収束していない / entropy collapse** が Phase 1 で解くべき課題。

## 既知のリスク / 過去知見

- **from-scratch RL は competent 相手にブートストラップ不可(天井)**(memory `case7_train_eval_parity_gap`)。
  → 直接 rulebase に当てず self-play/同実力で 0.5 を保ち、固定相手は測るだけ。
- **強すぎ相手の直接学習は reward 飽和で有害**(memory `unbeatable_opponent_harmful`)→ f_var で 0.5 維持。
- **本物 rulebase(python_v8)は sequential host-callback で JAX rollout 不可**、iter0 が GPU0% 停滞
  (memory `python_v8_train_pool_gpu_stall`)→ in-loop は in-JAX proxy、本物 case8 は offline 評価。
- **horizon は必ず 500**(memory `reinforce_horizon_terminal_reward_bug`)、長 run は iter ごと S3 upload。

## 実施しない検証 / 評価 (skip list)

### 評価
- Kaggle publicScore は引用しない (project rule, memory `project_om_finding` / `project_case5_validation`)
- skill rating は採否に使わない (project default)
- in-loop 採否は ① pool 勝率 ② 固定相手 held-out 勝率曲線 ③ 速度 の 3 指標で行う(本物 case8 300戦は最終候補のみ)

### 分析
- n<300 結果で結論を出さない(最終確認は本物 case8 と n≥300 paired)
- replay 詳細分析は実施しない(experiment-analysis は曲線/勝率集計のみ)

### 実行
- Kaggle submit / promote はしない(本ループ対象外、別途承認要)
- 各 RunPod run 後は pod を destroy(課金停止)
- smoke test (1-ep) は必須(skip しない)。oneshot 学習(`dev/runpod train`, auto-recover 有効)を既定とする

### 例外条件
- 本物 rulebase case8 (python_v8) との比較は学習ループ外の offline paired 300戦で行う(in-loop は in-JAX `baseline_jax_full` proxy で進捗測定)

## 仮説リスト (priority 順)

- [x] (P0) H0: **case8 scaffold** — `pipeline/reinforce/case8/` を case7 コピーで作成、`algo: ppo|vmpo`
  フラグ + PFSP/held-out/Elo 配線。**adopted (iter1)**: GPU で algo=ppo・f_var pool・held-out
  (baseline_jax_full)・Elo・S3 ckpt 動作確認。後続 Phase の土台。

### Phase 1 — PPO 実験条件の確定(土台。ここで凍結した条件を Phase 2/3 がそのまま使う)

- [ ] (P1) H1: **PPO 実験条件探索** — 概要: PPO で固定相手・pool 構成・iteration 数・LR 等の
  パラメータを変えて実験し、**適切な実験設定を確定**する。目的: ① pool 内 per-iter 勝率が ~0.5、
  ② 固定相手(in-JAX held-out)との勝率が **0 付近から滑らかに増加**、③ 全 JAX・GPU で
  **20 iter を ~30分で完了**、の 3 条件を同時に満たす設定を見つける。変更候補(1 つずつ A/B):
  pool 構成(lite/full/self の有無)、curriculum switch_iter、f_var の priority_p/ema、LR、
  entropy_coef、episodes_per_iter、iteration 数。**この H1 の出力(凍結 config)が Phase 2/3 の固定条件**。

### Phase 2 — V-MPO 実装と無調整比較(条件は H1 で凍結、algo のみ変更)

- [ ] (P2, depends on H1) H2: **V-MPO 実装 + 無調整 A/B** — 概要: V-MPO loss(top-half adv target ψ +
  温度 η dual + trust-region α、importance weight/entropy reg なし)を新規実装し、**H1 で確定した
  PPO と同一の実験条件**(固定相手・pool・iteration 数・LR 等すべて同じ)で `algo=vmpo` だけ変えて学習。
  目的: V-MPO を **チューニングなし(論文デフォルト HP)** で実行した場合、PPO に比べてどう振る舞うか
  検証(pool 勝率収束 / held-out 勝率曲線 / entropy 推移 / 速度 を PPO と並置)。
  **条件は一切変えない — 変えると PPO と比較不能**。

### Phase 3 — V-MPO のパラメータチューニング(条件は H1 で凍結、V-MPO 内部 HP のみ sweep)

- [ ] (P3, depends on H2) H3: **V-MPO HP チューニング** — 概要: **H1 で確定した PPO と同一の実験条件**で、
  V-MPO の内部 HP(ε_η / ε_α / top-k 割合(既定 0.5)/ η・α 初期値)を 1 つずつ sweep し最適値を選ぶ。
  目的: V-MPO のパラメータチューニング。各 sweep 点も実験条件は固定し、V-MPO HP のみを変える A/B。
  最良 V-MPO を H2 の PPO ベースラインと比較し、本物 case8 と offline paired 300戦で最終確認。

## Iteration log

(各 iter 完了時に experiment-analysis / experiment が追記)

| iter | 開始 | 仮説# | plan path | run_id | 主要メトリクス | 採否 | result path |
|---|---|---|---|---|---|---|---|
| 1 | 2026-06-09 | H0 | iter1_plan.md | 20260609-062331__feature-poc-v-mpo__0c08b8a__seed0 | scaffold動作: algo flag/PFSP/held-out(baseline_jax_full)/Elo GPU確認, held-out iter0=0.25 iter4=0.375 | adopted | iter1_result.md |

## 参考 (References)

- V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control
  (Song et al., DeepMind, ICLR 2020, arXiv:1909.12238) — 非パラメトリック target distribution を
  top-half advantage + 温度 η で構成 → 重み付き最尤で policy 更新 → decoupled KL trust-region(α)で制約。
  importance weighting / entropy reg / population-based tuning なしで Atari-57・DMLab-30 の従来 score を更新。
  policy collapse 耐性が PPO 比の主な売り。https://arxiv.org/abs/1909.12238
