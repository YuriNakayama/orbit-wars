# Reinforce/case5 — support_reward (iter4)

> 作成日: 2026-05-28
> 仮説 ID: H5 (production potential — ratio coef=1.0 base に重畳)
> hypotheses.md: docs/experiment/reinforce/20260527_case5_support_reward/hypotheses.md
> 関連: iter3_plan.md / iter3_result.md (H4 ratio coef=1.0: lite last-10 0.820, 現行最良)
> スコープ: 惑星保持割合を「production 加重」にした ratio_prod mode を追加し、H4 (coef=1.0) base で比較

## 仮説 (Hypothesis)
H4 の planet ratio は惑星を**等価**に数える (count ベース)。実際には高 production 惑星の保持が
戦略的に重い。planet 信号を **production 加重保持割合** `prod_mine/(prod_mine+prod_enemy)` に
置き換える (ship ratio はそのまま) と、領域支配の「質」を報酬に反映でき last-10 0.820 をさらに
押し上げる可能性がある。potential-based 維持 (turn 差分)、coef=1.0 据え置き。

## 既存コードの現状 (from Step 1)
- `EnvState.planet_prod` (int32[MAX_PLANETS]) が各惑星の production を保持 → 加重和に流用可。
- `rollout_jax.py`: `_shaping_potentials(state, seat, mode)` が mode 別に (Φ_ship, Φ_planet) を返す。
  ratio mode は ship/planet とも保持割合。2 本 carry の枠は変えずに planet 側を production 加重に
  差し替える新 mode `ratio_prod` を追加すれば carry 拡張不要。
- iter3 所見: ratio coef=1.0 が last-10 0.820、value_loss 0.0066 と安定。chunk まだ上昇中。

## スコープ (Scope)
- 変更ファイル:
  - `bot/pipeline/reinforce/case5/training/rollout_jax.py`
    - `_production_totals(state, seat) -> (prod_mine, prod_enemy)` 追加 (planet_prod を owner で集計)。
    - `SHAPING_MODE_RATIO_PROD = 4` 追加 (name_to_int `"ratio_prod"`)。
    - `_shaping_potentials` の ratio_prod 分岐: Φ_ship = ship 保持割合、
      Φ_planet = production 加重保持割合 `prod_mine/(prod_mine+prod_enemy+eps)`。
    - `_shaping_coefs`: ratio_prod は ratio と同じ `(shaping_coef, shaping_coef)`。
  - `bot/pipeline/reinforce/case5/configs/kaggle_jax_train_h5_ratio_prod.yaml` (新規、h4 base で shaping_mode=ratio_prod, coef=1.0)
  - `bot/src/gpu/runpod/config/cases.py` に stage 追加
- ハイパーパラメータ: `shaping_mode: ratio → ratio_prod`、`shaping_coef=1.0` (H4 据え置き)。
- データセット / 特徴量変更: なし。

## 実装ステップ (Implementation outline)
1. `rollout_jax.py`: `_production_totals` 追加。
2. `rollout_jax.py`: `SHAPING_MODE_RATIO_PROD=4` + name_to_int、`_shaping_potentials` に ratio_prod 分岐、
   `_shaping_coefs` の switch を 0..4 に拡張 (ratio_prod も (coef,coef))。既存 mode 非破壊。
3. 新 yaml `kaggle_jax_train_h5_ratio_prod.yaml` 作成。
4. `cases.py` に `reinforce_case5_kaggle_jax_train_h5_ratio_prod` stage 追加。
5. ユニットテスト: ratio_prod の係数解決 + production 加重 ratio が [0,1] + rollout 非発散 + 既存非破壊。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)
- ローカル self-play 300 対戦は行わない (学習中 last-10 win_rate + trend で採否)。
- Kaggle publicScore / skill rating 不使用。
- n<300 で確定判定しない (default ON) → win-rate は inconclusive 固定、trend で傾向判断。
- replay 分析は学習ログ base。
- 例外条件: なし。

### 実施する検証
- ローカル: `dev/test-bot` + `uv run --directory bot pytest tests/pipeline/reinforce/case5 -x`
- smoke: ratio_prod rollout で reward 有限・ratio∈[0,1] をユニットテストで担保。
- リモート: `dev/runpod train <sha> --case reinforce_case5_kaggle_jax_train_h5_ratio_prod --gpu-name "NVIDIA GeForce RTX 3090" --gpu-name "NVIDIA GeForce RTX 4090"` (3090/4090 限定)、~2.5h。
- 評価: 対戦相手 baseline_jax_lite (in-training)、128 ep/iter、主要メトリクス = lite phase last-10 win_rate + trend、
  採否しきい値 = H4 (ratio count, last-10 0.820) と比較し +3pp で production 加重採用、悪化なら H4 維持。

## リスク / 既知の不確実性
- production 加重は home planet 等の高 prod 惑星に報酬が偏る → 序盤の neutral 確保を軽視する副作用の懸念。
  trend / chunk 推移で確認。悪化なら count ベース (H4) が最良として確定。
- production 値域が planet count と異なるため ratio 正規化後も分布が変わる → value_loss 監視。
