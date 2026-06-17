# case7 Pool 形式 RL — Codebase Research

時刻基準: 2026-06-04 / 対象: `bot/pipeline/reinforce/case7/`

## Deep Codebase Analysis

### Area 1: opponent pool dispatch (rollout 側)
- **Files**: `training/rollout_jax.py:39-73`(import + mode 定数), `:385-540`(`_rollout_one_env` + `lax.switch`)
- **現状**: opponent は **7 mode の int** で表現され `jax.lax.switch(clip(mode,0,6), [...])` で分岐:
  `0 noop / 1 baseline_jax_lite / 2 baseline_jax_full / 3 self_snapshot / 4 python_v1 / 5 python_v4 / 6 python_v8`。
- **key interface**:
  - `_baseline_jax_actions(state, seat)` / `_baseline_jax_full_actions(state, seat)` — case1 の JAX rule、署名 `(EnvState, int) -> (L,3)`。**完全 in-JAX**(host hop 無し)。
  - `_self_snapshot_opponent_actions(opp_model, state, seat)` — 凍結 self を featurize→argmax。
  - `_python_v1/v4/v8_opponent_actions(state, seat)` — `pure_callback` で本物 Python(重い)。
- **patterns**: 全 opponent が同じ `(state, 1-seat) -> (L,3)` 署名に揃えられ、`lax.switch` で1つだけ選択。vmap/scan friendly に int mode を保持。
- **gap**: **mode は 0-6 にハードコード**。新 opponent(case8 JAX rule)を足すには (a) import (b) action fn 追加 (c) switch branch 追加 (d) `OPPONENT_NAME_TO_MODE` 追記 の4点。

### Area 2: host 側 pool 管理 (train 側)
- **Files**: `training/train_jax.py:440-540`(pool/selector class), `:600-780`(iter loop での dispatch)
- **`_OpponentPool(cap)`** (`:440`): frozen self snapshot の FIFO。`push(model)`(満杯で先頭drop)/`sample()`(uniform)。host-side Python。
- **`_OpponentEntry`** (`:470`): `opponent:str` + `model` + `win_ema:float`。1 選択肢を表す。
- **`_PrioritizedOpponentSelector(p, ema, init_win)`** (`:477`): entries = `[baseline_jax_full] + pool snapshots`。`select()` は重み `(1-win_ema)^p`(f_hard)で1つサンプル。`update(win_rate)` で EMA 更新。`rebuild()` で pool 入替時に win_ema を index 整合で carry-over。
- **iter loop** (`:600-780`): `switch_iter` 前は `early`(noop)、後は `pool`。`snapshot_every` 毎に現 model を pool へ push。各 late iter で selector が `self_snapshot` か `baseline_jax_full` を選び、`opponent_mode` int + `opp_model` を rollout に渡す。
- **gap1**: pool に混ぜられる「rule 相手」は **`baseline_jax_full` 固定**(`:480` のコメント)。case8 を足すには selector の entry 種別を拡張。
- **gap2**: `_save_best_pt` は毎 iter 無条件保存=「最新」。`win_rate>=best` は S3 upload gate のみ。→ iter15 で per-iter ckpt を追加したが **sweep は手動 shell**(`post_train_eval.sh`)、train ループ内に未統合。

### Area 3: case8 JAX rule agent (pool 追加候補)
- **Files**: `bot/pipeline/rulebase/case8/baseline_jax/agent_jax.py`, `world_features.py`
- **`compute_actions(features: WorldFeatures, modes: ModesArrays) -> (L,3)`** (`agent_jax.py:157`)、`compute_actions_jit` 版あり(`:174`)。
- **★決定的発見**: `world_features.py:700-` に **`build_world_features_from_state(state, seat)`** が存在。
  「pure-JAX path: 固定 shape EnvState から WorldFeatures を構築、jit/vmap 可能」と明記(`:700-704`)。
- **`_modes_from_features(features)`** (`agent_jax.py:55`) で modes 派生。
- **→ 含意**: case8 opponent action fn は
  `compute_actions(build_world_features_from_state(state, 1-seat), _modes_from_features(...))`
  だけで **完全 in-JAX**(`baseline_jax_full` と同じ扱い、host hop 不要)で書ける。**最大の技術的障壁が無い**。
- **caveat**: 署名が case1 full(`(state,seat)`)と異なるため、rollout 側に薄い adapter wrapper(`_baseline_case8_actions(state, seat)`)を1つ書いて switch に挿す。

### Area 4: shaping reward (case5 graft)
- **Files**: `rollout_jax.py` の `_shaping_potentials`/`_shaping_coefs`
- ratio PBRS `Φ=mine/(mine+enemy)` を ship/planet に適用。`shaping_mode: ratio` / `shaping_coef: 1.0` が case5 H4 最良。combined は係数次第で reward 爆発(iter11 で実証)。**変更不要、現状維持**。

### Area 5: 評価 (eval_vs_baseline)
- **Files**: `evaluation/eval_vs_baseline.py`、CHALLENGER=`rl_v7`。
- jax_to_torch で npz best.pt → torch weights.pt 変換後、本物 env で対戦。
- **gap**: train ループは self-play win しか見ず、**外部 eval(vs rl_v0/case8)が train から切れている**(手動 shell)。iter15 で「self-play win は外部汎化と無相関」を実証済 → model 選択を train 内に統合する価値。

## Technical Constraints
- **horizon は必ず 500**(memory `project_reinforce_horizon_terminal_reward_bug`)。<500 だと game(~497turn)未終了で terminal 報酬消失。
- **飽和相手は有害**(memory `project_reinforce_unbeatable_opponent_harmful`)。lite/full は win 0.17=ほぼ -2.0 飽和。混入率を低く保つか、f_hard で自然減衰させる。
- **self-play 回しすぎ overfit**(同 memory)。pool で相手分布を非飽和に保つのが PFSP の核心。
- **JAX self-play は foreground 実行必須**(memory `feedback_jax_selfplay_foreground_only`)。background sandbox で CPU hang。
- **self_snapshot は rollout を重くしない / python_v* は重い**(相手 forward 分)。case8 JAX rule も in-JAX なので軽い。
- **DVC cache 共有**: worktree 間同時 repro/pull は lock 競合。`data` は循環 symlink で path 解決注意。

## Key Findings Summary
1. **pool への case8 追加は低リスク**: `build_world_features_from_state` が既存 → in-JAX で `baseline_jax_full` と同列に挿せる。host hop 不要。
2. **selector が rule 相手を `baseline_jax_full` 固定**: case8 を entry 種別に足す host 側改修が必要(`_PrioritizedOpponentSelector`)。
3. **model 選択ロジックが train から切れている**: per-iter ckpt はあるが sweep は手動。train 内 in-JAX 軽量 eval(vs 固定 rl_v0/case8)を統合すれば「sweet spot 自動回収」が制度化できる。
4. **shaping/horizon は確定済み、変更不要**(ratio/1.0, horizon 500)。
5. **既知の天井**: 小規模で v1 0/10。case8(本物 parity)を pool に入れても飽和すれば有害。**混入率と f_hard 減衰の設計が成否を分ける**。
