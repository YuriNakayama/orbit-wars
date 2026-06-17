# case7 Pool 形式 RL — 要件定義

## 背景と目的
`pipeline/reinforce/case7/` で、**JAX 化ルールベース(case8 本物 parity)+ 過去自身 snapshot**
を opponent pool として用いる PFSP 強化学習を行う。既存 case7 は past-self pool +
lite/full + ratio shaping を実装済みだが、(1) pool に入る rule 相手が飽和気味の
lite/full 固定、(2) model 選択が手動 sweep で train から切れている。本機能で
**本物 parity の case8 を低率 exploiter として pool に追加**し、**model 選択を train 内 in-JAX
eval に制度化**する。小規模 CPU(~20min)で設計検証後、採否を見て GPU 段階拡大。

## User Stories(優先度順)
1. 学習者として、pool に case8(本物 parity)を低率混入させ、飽和を避けつつ
   「本物 v1 に近い学習相手」と対戦したい(SUMMARY が欠落と指摘した枠)。→ **最優先**
2. 学習者として、各 iter 末に固定相手と in-JAX 軽量 eval し、self-play win でなく
   外部勝率で best を自動選択したい(iter15 の手動 sweep を制度化)。→ **高**
3. 学習者として、case8 混入率を f_hard + 低上限キャップで制御し、飽和暴走を防ぎたい。→ **高**
4. 開発者として、散らかった 15 iter の config を pool 構成中心にクリーン整理したい。→ **中**
5. 学習者として、設計検証(CPU)で採用と出たら RunPod で iterations 拡大したい。→ **中**

## 機能要件
1. **case8 opponent mode 追加**: `_baseline_case8_actions(state, seat)` を rollout に追加し、
   `lax.switch` の新 branch(mode 7)+ `OPPONENT_NAME_TO_MODE["baseline_jax_case8"]` 登録。
   実装は `compute_actions(build_world_features_from_state(state, 1-seat), modes)`、完全 in-JAX。
2. **selector に case8 entry**: `_PrioritizedOpponentSelector` の固定 entry を
   `[baseline_jax_full]` → `[baseline_jax_full, baseline_jax_case8]`(config で有効化)に拡張。
   f_hard 重みは既存 `(1-win_ema)^p`。
3. **混入率キャップ**: case8(および full)の合計選択確率に上限(`exploiter_prob_cap`, 既定 0.2)。
   超過分は past-self snapshot に再配分。
4. **train 内 in-JAX eval**: 各 iter 末、固定相手(`eval_opponent`, 既定 `baseline_jax_case8` か
   `il_v0` 相当)と N=`eval_episodes` 戦を in-JAX で実行。`eval_win` を metrics に記録し、
   `eval_win >= best_eval_win` で best.pt を gate(self-play win でなく外部勝率で選択)。
5. **metrics 拡張**: 既存 row に `eval_win`, `eval_opponent`, `exploiter_sel_rate`(case8/full 実選択率)を追記。
6. **config 整理**: pool 構成を表現する単一の正典 config(`pool_default.yaml`)を作り、
   loop_iter* の散逸を README の表に集約。

## 非機能要件
- **性能**: case8 opponent は in-JAX(host hop 無し)で self_snapshot 同等の rollout コスト。
  in-JAX eval も同様に軽量(python_v* の `pure_callback` は使わない)。CPU ~20min/run を維持。
- **再現性**: seed 固定、horizon=500 厳守(memory: 必須)。
- **安全性**: GPU 起動は段階的、Kaggle submit/promote はユーザー承認必須。
- **保守性**: ruff/mypy clean、case 独立(他 case を import しない=case8 のコードは複製でなく
  `pipeline.rulebase.case8` 経由 import、ただし train/rollout は submission 対象外なので可)。

## スコープ外
- shaping reward の変更(ratio/1.0 確定、変更しない)。
- 新規 OSS 依存の追加。
- Kaggle 本番 submit(別途 `dev/submit`、承認制)。
- v1 越えの保証(小規模では構造的に不可と実証済、本機能は設計健全化と弱〜互角相手攻略が目標)。

## 用語
- **PFSP**: Prioritized Fictitious Self-Play。勝ちにくい相手を優先サンプル。
- **exploiter**: pool 内の rule 相手(full/case8)。past-self(main)と区別。
- **f_hard**: 優先度 `(1-win_ema)^p`。
- **in-JAX eval**: host hop 無しで EnvState を回す評価(本物 env でなく JAX sim)。
