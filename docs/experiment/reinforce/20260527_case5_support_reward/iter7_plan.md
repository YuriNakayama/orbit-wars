# Reinforce/case5 — support_reward (iter7)

> 作成日: 2026-05-31
> 仮説 ID: H6 (勝ちターン短縮 time bonus、H3 pair)
> hypotheses.md: docs/experiment/reinforce/20260527_case5_support_reward/hypotheses.md
> 関連: iter6_result.md (H3 dense rejected, PBRS必要性実証) / iter3_result.md (H4 ratio = 現行最良 0.820)
> スコープ: 早期勝利に terminal bonus / 引き伸ばしに小ペナルティを追加、H4 base に重畳

## 仮説 (Hypothesis)
H3 (絶対 dense) が引き伸ばし方向にバイアスしたことを受け、H6 は **時間方向の整形** で
バランスを取る:
- 勝利時 (terminal sign +1) に `time_bonus_coef · (1 - turn/horizon)` のボーナスを加算
  → 早く勝つほど報酬大
- 各 turn に `-time_penalty · ε` の小ペナルティ → 長引かせると累積で不利

H4 ratio (PBRS) base に重畳。time bonus は terminal 時のみで PBRS とは独立に動くので
PBRS 性質を破壊しない (terminal reward の整形)。H3 の引き伸ばしバイアスが顕在化した
からこそ、時間圧力が必要かを別軸で検証。

期待: 早期勝利の policy 改善で +3pp 採用、効果薄なら inconclusive。
※ H6 単独で大幅改善は見込み薄 (H4 既に 0.820)。試行的検証 (リスト消化最後)。

## 既存コードの現状 (from Step 1)
- `rollout_jax.py` step_fn: `terminal_reward = jnp.where(term, jnp.sign(r_self-r_opp), 0.0)` で
  ±1 のみ。time bonus は `term` 分岐内で `+ time_bonus * (1 - t/horizon)` を加算すれば実装可。
- per-turn time penalty: shaping_reward に `- time_penalty` を毎 turn 加算 (shaping_clip 前)。
- iter3-6 で H4 base が確立、time bonus はその上に重畳。

## スコープ (Scope)
- 変更ファイル:
  - `bot/pipeline/reinforce/case5/training/rollout_jax.py`
    - `_rollout_one_env` / `collect_rollout_jax` に `time_bonus_coef: float = 0.0`,
      `time_penalty_coef: float = 0.0` 引数追加。
    - step_fn 内、terminal_reward 算出時に勝利なら `time_bonus_coef * (1.0 - t / horizon)` を加算
      (t = scan の `_t` index)。
    - shaping reward に `-time_penalty_coef` を毎 turn 加算 (PBRS shaping の後、dense 加算の前、
      clip の前)。dense_coef=0 のとき H3 経路は無効化なので影響なし。
    - vmap in_axes を 2 本拡張 (None, None)。
  - `bot/pipeline/reinforce/case5/training/train_jax.py`: YAML から両 coef 読込→plumb、history 記録。
  - `bot/pipeline/reinforce/case5/configs/kaggle_jax_train_h6_time_bonus.yaml` (新規、h4 base +
    `time_bonus_coef: 0.5`, `time_penalty_coef: 0.001`)
  - `bot/src/gpu/runpod/config/cases.py` に `reinforce_case5_kaggle_jax_train_h6_time_bonus` stage 追加
- ハイパーパラメータ: H4 (ratio coef=1.0) 据え置き + `time_bonus_coef=0.5` (早期勝利で最大 +0.5)、
  `time_penalty_coef=0.001` (500 turns で累積 -0.5、勝てば time_bonus と相殺)。
- データセット / 特徴量変更: なし。

## 実装ステップ (Implementation outline)
1. `rollout_jax.py`: 2 引数追加、step_fn で `t` index を carry に追加、terminal_reward と
   shaping_reward に各 coef を適用。vmap in_axes 拡張。
2. `train_jax.py`: YAML 読込→`_run_iter`→rollout に plumb、history 記録。
3. 新 yaml `kaggle_jax_train_h6_time_bonus.yaml` (h4 base + time_bonus_coef=0.5, time_penalty_coef=0.001)。
4. `cases.py` に stage 追加。
5. ユニットテスト: time coef=0 で既存と bit-identical / time_bonus>0 で勝利時報酬が増 /
   time_penalty>0 で長期 turn cumulative が下がる。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)
- ローカル self-play 300 対戦は行わない。
- Kaggle publicScore / skill rating 不使用。
- n<300 で確定判定しない (default ON) → win-rate verdict fixed inconclusive。
- replay 分析は学習ログ base。

### 実施する検証
- ローカル: `dev/test-bot` + `uv run --directory bot pytest tests/unit/pipeline/reinforce/case5 -x`
- smoke: time coef=0 で非破壊 / time_bonus>0 で勝利時報酬増をユニットテストで担保。
- リモート: `dev/runpod train <sha> --case reinforce_case5_kaggle_jax_train_h6_time_bonus --gpu-name "NVIDIA GeForce RTX 4090" --gpu-name "NVIDIA GeForce RTX 3090" --cloud-type ALL` (**consumer 限定**)、~2.5h。
- 評価: lite phase last-10 + trend vs H4 0.820、採否しきい値 = H4 +3pp で採用、同等 inconclusive、劣化 rejected。
- 追加観点: H3 のような win_rate inflation が出ないかを value_loss が H4 並み (< 0.05) で確認。

## リスク / 既知の不確実性
- time_bonus は terminal 時のみで dense 加算ではないので H3 のような inflation は出ない見込み。
  ただし horizon=500 で勝利エピソードでも turn=300 平均 → time_bonus 平均 +0.2 程度の terminal 修正
  なので、policy 大幅変化は期待薄。
- time_penalty 0.001/turn × 500 = -0.5 は終局報酬と同オーダー、過剰なら勝てる試合を諦めるリスク。
  Trade-off 観点で小さめ設定。
- H6 で +3pp 達成は楽観的、現実的には inconclusive (H4 0.820 維持) が最有力見込み。
  最後の仮説消化で hypotheses.md 状態を completed に持っていくのが主目的。
