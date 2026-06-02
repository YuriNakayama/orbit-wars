# Reinforce/case5 — support_reward (iter2)

> 作成日: 2026-05-28
> 仮説 ID: H2 (保持「割合」差分 shaping)
> hypotheses.md: docs/experiment/reinforce/20260527_case5_support_reward/hypotheses.md
> 関連: iter1_plan.md / iter1_result.md (H1 combined: last-10 0.549 / trend +0.376, inconclusive)
> スコープ: rollout_jax に ratio shaping mode を追加 (potential = mine/(mine+enemy))

## 仮説 (Hypothesis)
絶対数の差分 (H1) ではなく **保持割合** `mine/(mine+enemy)` を potential とし、その turn 差分を
報酬 shaping にする。ship・planet それぞれ割合は [0,1] に正規化されるため production スケールに
依存せず、係数調整が容易で報酬スケールが安定する。potential-based を維持し最適方策をバイアスしない。

## 既存コードの現状 (from Step 1)
- `bot/pipeline/reinforce/case5/training/rollout_jax.py`:
  - `_ship_totals` / `_planet_count_totals` が (mine, enemy) を返す (割合算出に流用可)。
  - `_shaping_diffs` (H1 で追加) が (Δship, Δplanet) を返す。`_shaping_coefs` が mode 別係数を解決。
  - 現 mode: ships(0) / planets(1) / combined(2)。
- iter1 所見: combined は last-10 0.549 / trend +0.376 で baseline 比 +~5pp、ただし後半 0.55 頭打ち。
  ratio は分母正規化により後半の頭打ち改善 or 学習序盤の立ち上がり加速が期待できる。

## スコープ (Scope)
- 変更ファイル:
  - `bot/pipeline/reinforce/case5/training/rollout_jax.py`
    - `SHAPING_MODE_RATIO = 3` を追加 (name_to_int に `"ratio"`)。
    - ratio potential: `r_ship = mine_ships/(mine_ships+enemy_ships+eps)`、
      `r_plt = mine_plt/(mine_plt+enemy_plt+eps)` を算出する `_shaping_ratios(state, seat)` 追加。
    - ratio mode の reward = `shaping_coef * ((r_ship - prev_r_ship) + (r_plt - prev_r_plt))`。
      2 本独立 carry は H1 で導入済みの仕組みを ratio 値に流用 (carry を diff→generic potential に汎用化)。
  - `bot/pipeline/reinforce/case5/configs/kaggle_jax_train_h2_ratio.yaml` (新規)
    - kaggle_jax_train.yaml ベース、`shaping_mode: ratio`, `shaping_coef: 0.50`。
- ハイパーパラメータ / config: `shaping_mode: combined → ratio`、`shaping_coef=0.50` (ratios [0,1] に対し標準)。
- データセット / 特徴量変更: なし。

## 実装ステップ (Implementation outline)
1. `rollout_jax.py`: `SHAPING_MODE_RATIO` 定数 + name_to_int エントリ追加。
2. `rollout_jax.py`: `_shaping_ratios(state, seat) -> (r_ship, r_plt)` 追加 (eps=1e-6 でゼロ除算回避)。
3. `rollout_jax.py`: carry が持つ 2 本の potential を「diff or ratio」両対応にし、mode に応じて
   step reward を `c_ship·Δp_ship + c_planet·Δp_planet` (combined) か
   `shaping_coef·(Δr_ship + Δr_plt)` (ratio) に分岐。既存 ships/planets/combined は不変を保つ。
4. 新 yaml `kaggle_jax_train_h2_ratio.yaml` 作成。
5. `cases.py` に `reinforce_case5_kaggle_jax_train_h2_ratio` stage 追加。
6. 単体テスト: ratio mode の shaping reward が `shaping_coef·(Δr_ship+Δr_plt)` と一致、
   ratio が [0,1] に収まること、既存 mode の非破壊を確認。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)
- ローカル self-play 300 対戦は行わない (学習中 last-10 win_rate + reward trend で採否)。
- Kaggle publicScore は引用しない / skill rating も使わない (project rule)。
- n<300 結果で確定判定しない (default ON) → win-rate は inconclusive 固定、trend で傾向判断。
- replay 分析は実施する (学習ログ base、300対戦 dump は無し)。
- 例外条件: なし。

### 実施する検証
- ローカル: `dev/test-bot` + `uv run --directory bot pytest tests/pipeline/reinforce/case5 -x`
- smoke: ratio mode の rollout で reward 有限・ratio∈[0,1] を確認 (ユニットテストで担保)。
- リモート: `dev/runpod train <sha> --case reinforce_case5_kaggle_jax_train_h2_ratio`、~3h (RTX3090/4090, VRAM 24GB+)。
- 評価: 対戦相手 baseline_jax_lite (in-training)、128 ep/iter、主要メトリクス = lite phase last-10 win_rate + trend、
  採否しきい値 = H1 (last-10 0.549 / trend +0.376) と比較し +3pp or trend 明確改善で ratio 採用。

## リスク / 既知の不確実性
- ratio は序盤 (惑星少) で分母が小さく割合変動が大きい → spike 懸念。H7 (clip/正規化) が控えるが、
  本 iter では素の ratio で挙動を見る。spike が value_loss を荒らすなら H7 へ deepen。
- ship/planet ratio を等加算すると planet 信号が相対的に薄まる可能性 (H1 は planet 0.50 主体)。
  傾向次第で ratio の重み付けを follow-up 検討。
