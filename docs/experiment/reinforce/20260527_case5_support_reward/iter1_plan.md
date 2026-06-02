# Reinforce/case5 — support_reward (iter1)

> 作成日: 2026-05-27
> 仮説 ID: H1 (ship+planet 同時併用差分 shaping)
> hypotheses.md: docs/experiment/reinforce/20260527_case5_support_reward/hypotheses.md
> 関連: なし (本リスト初回 iter)
> スコープ: rollout_jax に combined shaping mode を追加し、ship 差分と planet 差分を個別係数で同時加算

## 仮説 (Hypothesis)
現状の shaping は `ships` か `planets` の **排他**二択 (`coef·Δ(mine−enemy)`)。
ship 差分と planet 差分を **同時併用** (`coef_ship·Δship + coef_planet·Δplanet`) すれば、
領域支配 (planet) と戦力 (ship) の両状態を密にフィードバックでき、収束速度・最終性能が
向上する。両項とも turn 差分なので potential-based を保ち最適方策をバイアスさせない。

## 既存コードの現状 (from Step 1)
- 主要モジュール:
  - `bot/pipeline/reinforce/case5/training/rollout_jax.py` — `_shaping_diff(state, seat, shaping_mode)` が `SHAPING_MODE_SHIPS` / `SHAPING_MODE_PLANETS` を `lax.cond` で排他選択。shaping = `shaping_coef * (diff - prev_diff)`。
  - `_ship_totals` / `_planet_count_totals` は既に実装済み (両方の (mine, enemy) を返す)。
  - `collect_rollout_jax(..., shaping_coef, shaping_mode)` → `_rollout_one_env` へ plumbing。
  - `training/train_jax.py` — config (`shaping_coef`, `shaping_mode`) を読み `_run_iter` → `collect_rollout_jax` へ渡す。
- 現 best レシピ: `configs/kaggle_jax_train.yaml` (`shaping_mode=planets`, `shaping_coef=0.50`, 200 iter, episodes=128)。
- 過去 iter の所見: なし (初回)。

## スコープ (Scope)
- 変更ファイル:
  - `bot/pipeline/reinforce/case5/training/rollout_jax.py`
    - `SHAPING_MODE_COMBINED = 2` を追加 (name_to_int に `"combined"`)。
    - `_shaping_diff` を combined 対応に拡張、または combined は別経路で `coef_ship·Δship + coef_planet·Δplanet` を直接算出。potential-based 維持のため **2 つの prev_diff** (ship/planet) を carry する設計が必要。
    - `_rollout_one_env` / `collect_rollout_jax` に `coef_ship`, `coef_planet` を追加 (combined 時のみ使用)。既存 `shaping_coef` の意味は ships/planets mode では不変。
  - `bot/pipeline/reinforce/case5/training/train_jax.py`
    - config から `coef_ship` / `coef_planet` を読み (default は combined 以外では未使用)、`_run_iter` → `collect_rollout_jax` へ plumbing。
  - `bot/pipeline/reinforce/case5/configs/kaggle_jax_train_h1_combined.yaml` (新規)
    - `kaggle_jax_train.yaml` をベースに `shaping_mode: combined`, `coef_planet: 0.50`, `coef_ship: 0.001` を設定。他は全継承。
- ハイパーパラメータ / config: `shaping_mode: planets → combined`、新規 `coef_planet=0.50` (= 現 best), `coef_ship=0.001`。
- データセット / 特徴量変更: なし。

## 実装ステップ (Implementation outline)
1. `rollout_jax.py`: `SHAPING_MODE_COMBINED` 定数と name_to_int エントリ追加。
2. `rollout_jax.py`: combined 用に ship-diff / planet-diff を **2 本独立** に carry し、`coef_ship·(Δship) + coef_planet·(Δplanet)` を step reward に加算する経路を実装 (既存 `_shaping_diff` 単一 diff 経路は ships/planets mode 用に温存)。
3. `rollout_jax.py`: `_rollout_one_env` / `collect_rollout_jax` に `coef_ship: float`, `coef_planet: float` 引数を追加 (vmap in_axes に None で broadcast)。
4. `train_jax.py`: `t_cfg.get("coef_ship", ...)` / `coef_planet` を読み `_run_iter` 経由で plumbing。
5. 新 yaml `kaggle_jax_train_h1_combined.yaml` を作成。
6. 単体テスト: `tests/pipeline/reinforce/case5/` に combined mode の shaping reward が `coef_ship·Δship + coef_planet·Δplanet` と一致する numerical テストを追加 (既存 ships/planets parity を壊さないことも確認)。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)
- ローカル self-play **300 対戦は行わない** — 学習中の last-10 iter 平均 win_rate (vs baseline_jax_lite, in-training) と reward trend で採否。
- Kaggle publicScore は引用しない (project rule)。skill rating も使わない。
- n<300 結果で結論を出さない (default ON)。
- replay 分析は実施する (補助、采否は学習メトリクス主体)。
- 例外条件: なし (H1 は potential-based、特例 300 対戦は不要)。

### 実施する検証
- ローカル: `dev/test-bot` + `uv run --directory bot pytest tests/pipeline/reinforce/case5 -x`
- smoke: 1-episode (もしくは数 iter) の JAX rollout smoke で combined reward が NaN/発散しないこと確認 (`configs/train_jax_smoke.yaml` ベースに combined を差した smoke、または `kaggle_jax_smoke.yaml`)。
- リモート: `dev/runpod train <commit-sha> --case case5` (config = `kaggle_jax_train_h1_combined.yaml`)、想定所要時間 ~3h (RTX 3090/4090, 200 iter)。VRAM 24GB+ 必須 (A4000 16GB 回避)。
- 評価: 対戦相手 baseline_jax_lite (in-training curriculum)、エピソード数 128/iter、主要メトリクス = last-10 win_rate + reward trend、採否しきい値 = baseline (planets 単体 0.50, last-10 ~0.50, trend +0.305) に対し **last-10 +3pp 以上 または trend 明確改善** で採用。

## リスク / 既知の不確実性
- ship 差分は production で絶対値が膨らみやすく、coef_ship=0.001 でも planet 項と桁が揃わない可能性 → value_loss/trend を見て H4 (係数 sweep) へ deepen 余地。
- combined で carry が増え (ship/planet 2 本の prev_diff)、jit trace・VRAM が微増。OOM 兆候は episodes_per_iter で調整。
- potential-based を厳密に保つには ship/planet それぞれ独立に Δ を取る必要がある (合算してから差分を取ると相互作用項で非 PBRS 化しうる) — 実装ステップ 2 で 2 本独立 carry を厳守。
