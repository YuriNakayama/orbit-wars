# Reinforce/case6 — PFSP self-snapshot opponent (iter1)

> 作成日: 2026-05-27
> 仮説 ID: H1 (P1)
> hypotheses.md: docs/experiment/reinforce/20260527_case6_pfsp/hypotheses.md
> 関連: なし (PFSP リスト最初の iter)
> スコープ: rollout_jax.py に self-snapshot opponent 経路を追加し、train_jax で curriculum late を self_snapshot に差し替え可能にする

## 仮説 (Hypothesis)

self-snapshot を 4 つ目の opponent モード (`self_snapshot`) として追加し、frozen な
自 param pytree を rollout に通して opponent seat の行動を自 agent の推論で埋める。
— これが全 self-play / PFSP 仮説 (H2/H4/H5/H6) の前提となる土台。本 iter 単体では
「過去 snapshot を相手にしても reward が崩れず学習が成立するか」を最小確認する。

## 既存コードの現状 (from Step 1)

- `training/rollout_jax.py`: opponent は **jit-trace 時の int コード** (`OPPONENT_NOOP`=0 /
  `BASELINE_JAX_LITE`=1 / `BASELINE_JAX_FULL`=2) で、`step_fn` 内 `jax.lax.switch` で分岐。
  rule 相手 (`_baseline_jax_actions` / `_baseline_jax_full_actions`) は **`state` のみ**から
  (L,3) 行を計算するので、`opponent_mode: int` を引数に通すだけで済んでいた。
- **self-snapshot は本質的に異なる**: frozen な model param pytree + opponent seat の
  featurize + policy forward + 決定論 action が必要。param は int コード化できないため、
  `_rollout_single` に **追加の `opp_model: ActorCriticJax | None` を closure/引数で渡す**形に拡張する。
- `training/train_jax.py`: `_opponent_for_iter(it)` が curriculum (early/late, switch_iter) を解決。
  snapshot は `_save_best_pt` で best.pt 保存。in-process では最新 model が毎 iter 参照可能。
- config `configs/train_jax.yaml`: `opponent: noop`、`opponent_curriculum` ブロックで early/late。

## スコープ (Scope)

- 変更ファイル:
  - `bot/pipeline/reinforce/case6/training/rollout_jax.py`
    — `OPPONENT_SELF_SNAPSHOT=3` 追加、`OPPONENT_NAME_TO_MODE` に `self_snapshot` 登録、
      `_rollout_single` / `collect_rollout_jax` に `opp_model: ActorCriticJax | None` 引数追加、
      `step_fn` で self_snapshot 時に opponent seat を featurize → `opp_model` forward →
      **決定論 (argmax) action** で (L,3) 行を生成し splice。
  - `bot/pipeline/reinforce/case6/training/train_jax.py`
    — `self_snapshot` 選択時に「学習開始時点の凍結コピー」を `opp_model` として渡す。
      本 iter は最小確認なので **iter0 snapshot 固定** (pool 化・更新は H2 の責務)。
  - `bot/pipeline/reinforce/case6/configs/` — smoke 用に `opponent: self_snapshot` の
    確認 config (既存 smoke を複製) を 1 本追加。
- ハイパーパラメータ / config: `opponent: noop → self_snapshot` (確認 config のみ)。
  iterations / episodes_per_iter / shaping は train_jax.yaml 既定を踏襲。
- データセット / 特徴量変更: なし。

## 実装ステップ (Implementation outline)

1. `rollout_jax.py`: `OPPONENT_SELF_SNAPSHOT=3` 定数 + `OPPONENT_NAME_TO_MODE["self_snapshot"]=3`。
2. `_rollout_single` / `collect_rollout_jax` シグネチャに `opp_model: ActorCriticJax | None=None` 追加。
   `opp_model is None` のとき従来挙動 (noop/lite/full) を完全保持。
3. `step_fn` 内: self_snapshot 時に opponent seat (`1-seat`) を `featurize_jax_w1` → `opp_model(batch)`
   → **argmax 決定論** で target_slot/ships を取り、`sampled_action_to_env_actions` で (L,3) 化。
   既存の `jax.lax.switch` を 4 分岐に拡張 (3 → self_snapshot lambda)。
   ※ opp_model を持たない (None) の場合 self_snapshot branch は noop にフォールバック。
4. `train_jax.py`: `self_snapshot` opponent 指定時、学習開始 model の凍結コピー (`jax.tree.map` で
   stop_gradient / detach) を生成し `opp_model` として `collect_rollout_jax` に渡す。
5. smoke 確認 config を `configs/` に追加 (`opponent: self_snapshot`、iterations 小)。
6. ユニットテスト: `tests/pipeline/reinforce/case6/` に self_snapshot rollout が
   shape 不変 + None 時に従来と一致することを確認するテストを追加。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)
- **ローカル self-play 300 対戦は行わない** — 採否は ① 学習 reward trend / last-10 mean を主軸とし、
  ② 100 戦 self-play・③ baseline_v1 との 20 戦は **方向性の参考値** (n<300 で結論を出さない、default ON)。
- Kaggle publicScore / skill rating は引用しない (project rule)。

### 実施する検証
- ローカル: `dev/test-bot` + `uv run --directory bot pytest tests/pipeline/reinforce/case6 -x`。
- smoke (必須): 1-ep self-play 相当 = `opponent: self_snapshot` の smoke config で
  `python -m pipeline.reinforce.case6.training.train_jax --config configs/<smoke>.yaml` が
  6-iter 完走し reward が NaN/発散しないことを確認。
- リモート: `dev/runpod train <commit> --case case6` (24GB+ VRAM = RTX 3090/A6000 を選択。
  A4000 16GB は OOM 実績 memory `project_runpod_a4000_oom`)。想定所要 ~1h (200 iter 規模)。
  ⚠️ best.pt 喪失 race / JAX npz bug は修正済 (commit c0cd427) だが pull 時に S3 fallback 確認。
- 評価: 対戦相手 = ① 学習 reward trend (主軸)、② vs 初期 snapshot 100 戦、③ vs baseline_v1 20 戦 (参考)。
  採否しきい値: reward trend が iter0-snapshot 相手で右肩上がり (trend > 0) かつ
  既存 noop/lite 経路の回帰なし (None フォールバックで従来テスト pass)。
- 分析: replay 分析 (experiment-analysis) を実施 (skip 指定なし)。

## リスク / 既知の不確実性

- **jit cache 増加**: opp_model を closure 経由で渡すと trace が分岐し再コンパイル / VRAM 増の懸念。
  None フォールバックと branch 分離でコンパイル数を最小化する。
- **opponent 決定論 vs sampling**: 本 iter は argmax 決定論で固定。sampling 相手の是非は H2 以降。
- **iter0 固定 snapshot**: 学習が進むと相手が弱すぎて reward が飽和する可能性 → pool 更新は H2 で対処。
