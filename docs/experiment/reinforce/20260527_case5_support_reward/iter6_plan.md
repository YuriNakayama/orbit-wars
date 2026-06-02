# Reinforce/case5 — support_reward (iter6)

> 作成日: 2026-05-30
> 仮説 ID: H3 (絶対保持数の非差分 dense 加算 — 対照群)
> hypotheses.md: docs/experiment/reinforce/20260527_case5_support_reward/hypotheses.md
> 関連: iter5_result.md (H7 inconclusive, clip 不要を確認) / iter3_result.md (H4 ratio coef=1.0 = 現行最良 0.820)
> スコープ: PBRS でない絶対保持数の dense 加算 (`coef · mine_count` 毎 turn) を H4 base に重畳、PBRS 系との性能差を対照確認

## 仮説 (Hypothesis)
H1〜H5/H7 は全て potential-based shaping (Δ(Φ_mine - Φ_enemy) または Δ ratio)。H3 は **non-PBRS**
として `coef · mine_count` (ships or planets) を **毎 turn 加算** する。これは Ng et al. 1999 が
警告する非 potential 加算であり、最適方策をバイアスさせる (貯め込み・引き伸ばし)。期待は **H4 比で劣化**、
PBRS の必要性を実証する対照群。劣化が明確なら早期 rejected (deepen しない、例外条件適用)。

## 既存コードの現状 (from Step 1)
- `rollout_jax.py`: `_shaping_potentials` が mode 別に (Φ_ship, Φ_planet) を返し、`_shaping_coefs`
  が (c_ship, c_planet) を返す。reward = `c·(Φ_now - Φ_prev)` で必ず Δ 化されている。
- H3 を実装するには Δ化を bypass する必要がある → 新 `dense_count` mode 追加、step_fn 内で
  `dense_reward = c_dense · mine_count` を Δ shaping に **加算** (Δ ではない素の量)。
  既存 mode (ships/planets/combined/ratio/ratio_prod) は非破壊で維持。
- iter5 所見: H4 が現行最良。H3 は対照のため H4 base に dense 加算を重畳する形が望ましい
  (H4 mode=ratio + dense_count 追加加算)。

## スコープ (Scope)
- 変更ファイル:
  - `bot/pipeline/reinforce/case5/training/rollout_jax.py`
    - `_mine_count_totals(state, seat) -> (ship_mine_count, plt_mine_count)` (planets と ships の自分側総量)。
    - `_dense_addition(state, seat, dense_coef_ship, dense_coef_planet) -> jax.Array`: 自分側の絶対量に係数を乗じた dense 加算項を返す。
    - `_rollout_one_env` に `dense_coef_ship`, `dense_coef_planet` 引数追加。step_fn の shaping
      計算後に `shaping = shaping + dense_addition` を加える。clip は dense 加算の前に適用 (PBRS 系の振る舞いを守る)。
    - `collect_rollout_jax` に同 kwarg、vmap in_axes を 2 本拡張。
  - `bot/pipeline/reinforce/case5/training/train_jax.py`
    - YAML から `dense_coef_ship` / `dense_coef_planet` 読込→`_run_iter`→rollout に plumb、history 記録。
  - `bot/pipeline/reinforce/case5/configs/kaggle_jax_train_h3_dense.yaml` (新規、h4 base + `dense_coef_ship: 0.01, dense_coef_planet: 0.1`)
  - `bot/src/gpu/runpod/config/cases.py` に `reinforce_case5_kaggle_jax_train_h3_dense` stage 追加
- ハイパーパラメータ: `shaping_mode=ratio` / `shaping_coef=1.0` (H4 据え置き) / `shaping_clip=0.0` (clip なし)
  + `dense_coef_ship=0.01`, `dense_coef_planet=0.1`。後者は planet=数個オーダー × 0.1 ≈ 0.5〜1/turn の追加加算で、
  PBRS shaping (Δ ≈ 0.01〜0.1/turn) と同オーダー〜やや大、引き伸ばしバイアスを観察しやすい強度。
- データセット / 特徴量変更: なし。

## 実装ステップ (Implementation outline)
1. `rollout_jax.py`: `_mine_count_totals` / `_dense_addition` 追加。
2. `_rollout_one_env` / `collect_rollout_jax` に `dense_coef_ship`/`dense_coef_planet` 引数追加、
   vmap in_axes 拡張、shaping に加算。
3. `train_jax.py`: YAML 読込→plumb→history 記録。
4. 新 yaml `kaggle_jax_train_h3_dense.yaml` (h4 全コピー + dense_coef_ship=0.01, dense_coef_planet=0.1)。
5. `cases.py` に stage 追加。
6. ユニットテスト: dense_coef=0 で既存 ratio と bit-identical (非破壊) / dense_coef>0 で報酬が dense 分上振れ /
   rollout 非発散。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)
- ローカル self-play 300 対戦は行わない (学習中 last-10 win_rate + trend で採否)。
- Kaggle publicScore / skill rating 不使用。
- n<300 で確定判定しない (default ON) → win-rate 単独は inconclusive 固定、trend で傾向判断。
- replay 分析は学習ログ base。
- **例外条件 (hypotheses.md より)**: H3 は明確に劣化/引き伸ばし傾向が出た場合は **rejected** として deepen しない。

### 実施する検証
- ローカル: `dev/test-bot` + `uv run --directory bot pytest tests/unit/pipeline/reinforce/case5 -x`
- smoke: dense=0 で非破壊 / dense>0 で報酬が dense 分上振れをユニットテストで担保。
- リモート: `dev/runpod train <sha> --case reinforce_case5_kaggle_jax_train_h3_dense --gpu-name "NVIDIA GeForce RTX 4090" --gpu-name "NVIDIA GeForce RTX 3090" --cloud-type ALL` (**consumer 限定、A100 不可**)、~2.5h。
- 評価: 対戦相手 baseline_jax_lite (in-training)、128 ep/iter、主要メトリクス = lite phase last-10 win_rate + trend、
  採否しきい値 = H4 (0.820) と比較し +3pp で dense 採用、悪化なら rejected (PBRS 必要性確認)、同等は inconclusive。

## リスク / 既知の不確実性
- dense 加算が大きすぎる (coef_planet=0.1×6 planets=0.6/turn) と、shaping 報酬 (0.01〜0.1) を埋もれさせる可能性。
  逆に小さすぎると効果が見えない。両極端なら 0.01/0.1 でも適切な信号差が出るはず。
- 引き伸ばしバイアスは horizon=500 turn 内で明確に出ない可能性 (lite phase の baseline_jax_lite 相手は
  通常 200-300 turn で終結)。trend の右肩上がりが H4 と乖離するかで判断。
- A100 は使わない (RunPod consumer 復活待ち、復活していなければスケジュール再起動を継続)。
