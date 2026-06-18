# Reinforce/case8 — pfsp_vmpo_vs_ppo (iter1)

> 作成日: 2026-06-09
> 仮説 ID: H0 (case8 scaffold)
> hypotheses.md: docs/experiment/reinforce/20260609_case8_pfsp_vmpo_vs_ppo/hypotheses.md
> 関連: docs/experiment/reinforce/20260608_case7_beat_rulebase/ (case7 PFSP+held-out+Elo の前身)
> スコープ: case7 を case8 へコピー + `algo: ppo|vmpo` フラグの配線 + held-out=case8/pool=case1+case8 化
> モード: list-consume (priority P1, topmost, 依存なし)

## 仮説 (Hypothesis)

`pipeline/reinforce/case8/` を case7 コピーで作成し、`training/train_jax.py` に `algo: ppo|vmpo`
フラグを通す。PFSP `f_var` / held-out / Elo の wiring を case8 向け(held-out=`baseline_v8`,
pool=`[case1 lite/full, python_v8, self-snapshot]`)に組み替える。
— これが成立すれば V-MPO/PPO を**同一 rollout・同一 PFSP・同一 held-out** で公平に A/B
する土台ができ、後続 H1-H5 が全て同じ harness 上に乗る。

## 既存コードの現状 (from Step 1)

- `pipeline/reinforce/case7/`(9,117 LOC, py)が PFSP+held-out+Elo の最新実装。case8 はこれを丸ごとコピー。
- `training/ppo_jax.py`: clipped surrogate + value MSE + entropy + BC-KL。`evaluate_actions_jax` /
  `make_optimizer` / GAE は **algo 非依存**で再利用可。`ppo_update_jax` が loss 本体。
- `training/train_jax.py`: L446 で `_ppo_update_jit(...)` を**ハードコード呼び出し**。`algo` フラグは未実装
  → ここに分岐を入れる(H0 ではフラグ配線のみ、vmpo は H1 で実装)。
- `training/rollout_jax.py`: opponent dispatch に `noop/lite/full/self_snapshot/python_v1/v4/v8` 実装済み。
  **`python_v8` (= rulebase case8) が pool・held-out 双方で即利用可**(`OPPONENT_PYTHON_V8=6`)。
- `training/train_jax.py` の `_PoolSelector` (`f_var`/`f_hard`)、`_heldout_eval`(YAML `heldout_eval.opponent`
  で指定可, 現状 default `baseline_jax_full`)、`_elo_update`(固定 ref anchor)。
- launch: `pipeline/reinforce/case7/training/launch_poc.sh` が dev/runpod dev pod + tmux で
  `python -m pipeline.reinforce.case7.training.train_jax --config ...` を起動。
  RunPod case 登録は `bot/src/gpu/runpod/config/cases.py` の `reinforce_case<N>_kaggle_jax_*` キー。
- 過去 iter 所見(case7): PPO+PFSP+dense+BC を尽くしても vs baseline_v8 = 0/10 が天井
  (memory `case7_train_eval_parity_gap`)。本 case は「rulebase に直接当てず self-play 0.5 + held-out 測定」で天井回避を狙う枠組み。

## スコープ (Scope)

- 変更ファイル(新規, case7 コピー起点):
  - `bot/pipeline/reinforce/case8/`(全体コピー。`case7` 参照を `case8` へ一括置換)
  - `bot/pipeline/reinforce/case8/training/train_jax.py` — `algo` フラグ分岐を追加(`ppo` は既存呼び出し, `vmpo` は H1 まで `NotImplementedError` か PPO fallback)
  - `bot/pipeline/reinforce/case8/training/ppo_jax.py` — そのまま流用(改変なし)
  - `bot/pipeline/reinforce/case8/configs/` — case8 用 config を新設(下記)
  - `bot/src/gpu/runpod/config/cases.py` — `reinforce_case8_kaggle_jax_train` / `_smoke` エントリ追加(`train_module=pipeline.reinforce.case8.training.train_jax`)
  - `bot/pipeline/reinforce/case8/training/launch_poc.sh` — case7 版を `case8` 置換
- config 変更(held-out / pool を case8 化, 出発点は case7 `h6_fvar3pool.yaml`):
  - `training.algo: ppo`(H0 既定)
  - `training.heldout_eval.opponent: python_v8`(← `baseline_jax_full` から変更)
  - PFSP pool: `priority: f_var`, `include_lite: true`, pool に `python_v8` を含める
  - `iterations: 10-20`(H0 短 run), `episodes_per_iter: 32`, `horizon: 500`
- データセット / 特徴量変更: なし(case7 featurizer をそのまま継承)

## 実装ステップ (Implementation outline)

1. `cp -r bot/pipeline/reinforce/case7 bot/pipeline/reinforce/case8` → 内部の `case7` import パス・docstring を `case8` へ一括置換(`pipeline.reinforce.case7` → `case8`)。
2. `case8/training/train_jax.py`: config に `algo` キーを追加し、L446 の `_ppo_update_jit` 呼び出しを `if algo=="vmpo": ...(H1) else: _ppo_update_jit(...)` に分岐。H0 では vmpo 分岐は `raise NotImplementedError("V-MPO is H1")` で明示。
3. `case8/configs/h0_smoke.yaml`(短 smoke, iterations=2)と `h0_ppo_short.yaml`(iterations=10-20, held-out=python_v8, f_var 3-opp pool)を新設。
4. `case8/training/rollout_jax.py`: pool entries / held-out が `python_v8` を受けられるか確認(既存実装で対応済みのはず → 変更不要なら確認のみ)。
5. `bot/src/gpu/runpod/config/cases.py` に `reinforce_case8_kaggle_jax_train` / `reinforce_case8_kaggle_jax_smoke` を追加。
6. `case8/training/launch_poc.sh` を case8 化(`--case case8`, train_module を case8 に)。
7. ローカル smoke(foreground 4-game gate)→ `dev/test-bot` → algo=ppo で 1-ep self-play smoke。
8. RunPod 短 run(algo=ppo, ~10-20 iter)で held-out=python_v8 / f_var pool の wiring が本番環境で動くこと(metrics.json に heldout_win_rate / elo / per-iter win_rate ~0.5 が出る)を確認 → 完了で pod destroy。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)
- ローカル self-play 300 対戦は行わない(採否は paired 30-60 戦 + 学習曲線 trend)
- replay 詳細分析は実施しない(experiment-analysis は勝率集計のみの skip モード)
- Kaggle publicScore は引用しない(project rule)/ skill rating は採否に使わない
- n<300 結果で結論を出さない(H0 は scaffold 成立=配線検証であり勝率採否はしない)

### 実施する検証
- ローカル: foreground 4-game gate(JAX self-play hang 対策)→ `dev/test-bot`(format/lint/type/pytest)→ `uv run --directory bot python -m pipeline.reinforce.case8.training.train_jax --config .../h0_smoke.yaml`(algo=ppo, 2-iter smoke)が完走
- smoke 必須(skip しない)。algo=vmpo は H0 では NotImplementedError で配線のみ確認(H1 で実装)
- リモート: `dev/runpod dev <sha> --case case8` → `launch_poc.sh <run_id> h0_ppo_short.yaml`、想定 ~10-20 iter / ~15-25min / ~$0.5
- 評価: H0 は**勝率採否なし**。受入条件 = ①両 algo でフラグ配線が通る(ppo 完走, vmpo は明示 NotImplemented)②RunPod 短 run で metrics.json に held-out vs python_v8 勝率 + Elo + per-iter win_rate が記録され、per-iter win_rate が f_var により ~0.5 近傍に収束する兆候があること
- run 完了後は pod を destroy(課金停止)

## リスク / 既知の不確実性
- case7 → case8 コピー時の import パス置換漏れ(`pipeline.reinforce.case7` 残存)で smoke が落ちる可能性 → grep で全置換確認。
- `python_v8` host callback が pool・held-out 双方で発火すると rollout 律速(memory `rollout_host_callback_bottleneck`)。held-out は別 rollout なので影響限定的だが、pool に python_v8 を含めると train rollout が重くなる可能性 → 必要なら pool は lite/full/self 中心にし python_v8 は held-out 専用に寄せる判断を H2 で行う。
- f_var の per-iter win_rate が ~0.5 に乗らない既知事象(case7 H5: ~0.29)。case7 H6 の count-based win_ema + include_lite 修正をコピー時に確実に引き継ぐ。
