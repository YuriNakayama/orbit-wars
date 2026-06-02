# Reinforce/case6 — PFSP snapshot pool + periodic refresh (iter2)

> 作成日: 2026-05-28
> 仮説 ID: H2 (P1, depends on H1)
> hypotheses.md: docs/experiment/reinforce/20260527_case6_pfsp/hypotheses.md
> 関連: iter1_plan.md / iter1_result.md / iter1_analysis.md
> スコープ: train_jax で snapshot を K iter ごとに pool 追加し、late 相手を pool + baseline_jax_full から選択

## 仮説 (Hypothesis)

K iter ごとに最新 snapshot を opponent pool に追加し、late 相手を「pool からの
ランダム抽出 + baseline_jax_full」から選ぶ。— H1 では frozen iter0 相手が surpass
後に信号枯渇 (iter150 で win~1.0 飽和)。pool を学習に追従させ相手を更新し続ければ、
win_rate が中間域 (0.5-0.7) に留まり reward が意味を持つはず。これが PFSP の前提。

## 既存コードの現状 (from Step 1 / iter1)

- `training/rollout_jax.py`: H1 で `OPPONENT_SELF_SNAPSHOT=3` + `opp_model` 引数を追加済。
  `collect_rollout_jax(..., opp_model=...)` で frozen snapshot を相手にできる。
  ただし opp_model は **単一 model** で固定 (vmap broadcast、scan trace 1 本)。
- `training/train_jax.py`: H1 で `opp_snapshot = model` (iter0 固定) を `_run_iter` に thread。
  pool 化されておらず周期更新もない。
- iter1 所見: iter5 switch で win 0.984→0.766 dip → snapshot は短期的に学習圧を出す。
  pool 更新で持続させるのが H2 の核心 (iter1_analysis.md)。

## スコープ (Scope)

- 変更ファイル:
  - `bot/pipeline/reinforce/case6/training/train_jax.py`
    — snapshot pool (list[ActorCriticJax], cap N=5) を保持。K iter ごとに現 model の
      凍結コピーを push (cap 超過で FIFO drop)。late iter では pool から 1 つランダム選択
      (host 側 Python で選び、その model を `opp_model` として渡す = trace は 1 本維持)。
      baseline_jax_full も late 候補に確率混在 (例: pool:full = 各 iter 50/50)。
  - `bot/pipeline/reinforce/case6/configs/kaggle_jax_train_h2.yaml` (新規)
    — H1 config 複製 + pool 設定 + **コスト軽量化** (iterations 100 / episodes_per_iter 64)。
  - `bot/src/gpu/runpod/config/cases.py` — `reinforce_case6_kaggle_jax_train_h2` stage 登録。
- ハイパーパラメータ / config:
  - **iterations: 200 → 100** (コスト半減、memory project_reinforce_self_snapshot_cost)
  - **episodes_per_iter: 128 → 64** (rollout 2倍重を相殺、~$2-3 目標)
  - 新規: `opponent_pool: {snapshot_every: 10, cap: 5, late_full_prob: 0.5}`
  - opponent=curriculum, switch_iter=5, early=noop, late=pool (新 late キーワード)
- データセット / 特徴量変更: なし。

## 実装ステップ (Implementation outline)

1. `train_jax.py`: `_OpponentPool` ヘルパ (host 側 list、push/sample)。K=snapshot_every
   ごとに `eqx` 凍結コピーを push、cap で FIFO。
2. late iter の opponent 解決: `late=pool` のとき、各 iter で pool.sample() か
   baseline_jax_full かを `late_full_prob` で確率選択し、選んだ model/mode を `_run_iter` へ。
   pool が空 (K 未到達) の間は iter0 snapshot で代替。
3. `rollout_jax.py`: 変更最小。opp_model 経路は H1 のまま流用 (pool 選択は host 側完結)。
   baseline_jax_full は既存 `OPPONENT_BASELINE_JAX_FULL=2` モードを使用。
4. config `kaggle_jax_train_h2.yaml` を iterations=100 / episodes=64 で新規作成。
5. cases.py に H2 stage 登録。
6. テスト: `tests/unit/.../case6/` に pool push/sample/cap の単体テスト追加。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)
- **ローカル self-play 300 対戦は行わない** — 採否は ① win_rate が 1.0 飽和せず中間域に
  留まるか、② reward/value_loss の推移を主軸。100 戦・20 戦 vs baseline_v1 は参考値。
- Kaggle publicScore / skill rating は引用しない (project rule)。n<300 で結論を出さない。

### 実施する検証
- ローカル: `dev/test-bot` + `uv run --directory bot pytest tests/pipeline/reinforce/case6 -x`
  + `tests/unit/pipeline/reinforce/case6 -x` (pool 単体テスト)。
- smoke (必須): pool 設定の smoke config で 6-iter 完走、pool push/sample が発火し
  reward NaN なしを確認。
- リモート: `dev/runpod train <commit> --case reinforce_case6_kaggle_jax_train_h2`。
  **iterations 100 / episodes 64 で想定 ~$2-3 (A100 fallback 時)**。
  ⚠️ uptime ベースで手動 cost 監視 (est_total=$0.000 バグ、memory)。cap 接近で手動 stop。
  3090/4090 が空けば優先 ($0.46-0.69/h)。
- 評価: 主軸 = win_rate が中間域維持か + entropy が収束に転じるか。
  ② vs 初期 snapshot 100 戦・③ vs baseline_v1 20 戦は方向性参考値。
- 分析: replay 分析は実施 (ただし JAX rollout は in-memory なので metrics 主体の skip mode)。

## リスク / 既知の不確実性

- **コスト再超過**: 軽量化しても A100 fallback で長引く可能性。uptime 手動監視必須。
- **pool sample の trace**: host 側で model を選べば scan trace は 1 本維持できるが、
  毎 iter 異なる pytree を渡すと re-trace の懸念 → 構造同一なら weights 差し替えのみで OK。
- **win_rate が依然飽和**: pool 更新頻度 K が大きすぎると相手が追従せず H1 と同じ飽和。
  K=10 で不足なら H7 (sweep) で詰める。
