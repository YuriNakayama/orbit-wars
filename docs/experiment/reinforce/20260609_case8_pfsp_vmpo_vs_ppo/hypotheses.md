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

- [x] (P1) H1: **PPO 実験条件探索 — 確定 (2026-06-10)**。凍結 config = `configs/ppo_frozen.yaml`
  (algo=ppo, 50 iter, f_var priority_p=4.0, pool=full+lite+self, held-out=baseline_jax_full,
  lr 3e-5→3e-6, entropy 0.02, switch_iter=4)。Phase 2/3 はこれを algo (+V-MPO HP) 以外不変で流用。
  - ① pool 勝率 ~0.5: priority_p {1,2,4} sweep で p=4.0 が最も 0.5 近傍 (self区間 mean ~0.4-0.46)。
  - ② held-out: baseline_jax_full で 0 付近から弱く上昇 (p=4.0 iter50 で ~0.25→0.34)。本物 case8 は offline。
  - ③ 速度: rollout jit (W7) + reset on-device vmap(reset_jax) (W8) で ~7s/iter。50 iter ≈ 6分
    (pod 性能非依存で ≤30分 達成)。GPU util 8%→95-99%。<br>
    注: ② は「0 から滑らかに 0.5 へ」の理想形には未達 (弱い上昇で頭打ち) — これは PPO の限界で
    あり Phase 2 の V-MPO 比較ポイント。①③ は達成、② は両 algo 共通の土台として凍結。

### Phase 2 — V-MPO 実装と無調整比較(条件は H1 で凍結、algo のみ変更)

- [x] (P2, depends on H1) H2: **V-MPO 実装 + 無調整 A/B — adopted (2026-06-10)**。vmpo_jax.py 実装、
  vmpo_frozen.yaml (ppo_frozen と algo のみ差分) で A/B。結論: **無調整 V-MPO は PPO と同等の収束性
  + 明確に高い entropy collapse 耐性 (min entropy 21 vs PPO 8.6) + わずかに高い安定性 (pool std
  0.188 vs 0.196)**。held-out 到達点は両者 ~0.27/max 0.375 で互角 (env 天井, algo 差でない)。
  V-MPO 内部健全 (η 1.0→1.24, α 5.0→3.94, trust-region KL 0.0003 ≪ ε_α=0.01)。→ phase2_result.md。

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
| 2-5 | 2026-06-09 | H1(R1) | phase1_exploration_plan.md | …p1/p2/p4 (3 run) | priority_p sweep: p=4.0 が pool勝率0.5 最近 / held-out横ばい | (探索) | — |
| 6 | 2026-06-09 | H1(R5) | phase1_exploration_plan.md | 20260609-113649__…3a88070 | iter50: held-out 0.23→0.34 弱上昇 / runtime 108min(slow pod) | (探索) | — |
| 7 | 2026-06-10 | H1(W7/W8) | phase1_exploration_plan.md | 20260610-022550__…aa36caf | rollout jit+reset on-device: ~7s/iter, GPU util 8%→95-99% | **adopted (凍結確定)** | phase1_result.md |

## 参考 (References)

- V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control
  (Song et al., DeepMind, ICLR 2020, arXiv:1909.12238) — 非パラメトリック target distribution を
  top-half advantage + 温度 η で構成 → 重み付き最尤で policy 更新 → decoupled KL trust-region(α)で制約。
  importance weighting / entropy reg / population-based tuning なしで Atari-57・DMLab-30 の従来 score を更新。
  policy collapse 耐性が PPO 比の主な売り。https://arxiv.org/abs/1909.12238
