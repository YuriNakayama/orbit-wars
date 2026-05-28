# Reinforce/case5 — support_reward (iter5)

> 作成日: 2026-05-29
> 仮説 ID: H7 (保持割合差分の clip / 正規化 — H2/H4 ratio 派生)
> hypotheses.md: docs/experiment/reinforce/20260527_case5_support_reward/hypotheses.md
> 関連: iter3_plan.md / iter3_result.md (H4 ratio coef=1.0: lite last-10 0.820, 現行最良) / iter4_result.md (H5 rejected)
> スコープ: ratio shaping の per-turn 報酬を band clip し序盤 spike を抑制、H4 (coef=1.0) base で比較

## 仮説 (Hypothesis)
H4 の ratio shaping は `mine/(mine+enemy)` ∈ [0,1] で正規化済だが、**序盤 (総 ship/惑星数が
小さい時)** は 1 機の増減で割合が 0→1 級に振れ、ΔΦ が大きな spike を生む。この spike が
advantage を荒らし value_loss/学習を不安定化させる懸念がある。per-turn shaping 報酬を
`[-clip, +clip]` に band clip すると、序盤の過大 spike を抑えつつ通常域の signal は素通し
でき、value_loss 安定化と last-10 のさらなる押し上げが見込める。H4 (ratio, coef=1.0) 据え置きで
clip=0.1 を重畳。

## 既存コードの現状 (from Step 1)
- `rollout_jax.py`: per-turn shaping = `c_ship·(ΔΦ_ship) + c_planet·(ΔΦ_planet)` を `step_fn` 内で算出
  (line 372 付近)。ratio mode では Φ∈[0,1] なので ΔΦ∈[-1,1]、coef=1.0 倍で報酬も同域。
- iter3 所見: ratio coef=1.0 が last-10 0.820、value_loss 0.0066、approx_kl 0.005 と既に安定。
  → H7 は「既に安定なので効果薄」の事前見込み (hypotheses.md 記載どおり)。本 iter はそれを実証する。
- `train_jax.py`: `shaping_coef` / `shaping_mode` / `coef_ship` / `coef_planet` を YAML→`_run_iter`→
  `collect_rollout_jax` に plumb 済。新 param `shaping_clip` を同経路に通せば carry 拡張不要。

## スコープ (Scope)
- 変更ファイル:
  - `bot/pipeline/reinforce/case5/training/rollout_jax.py`
    - `_rollout_one_env` / `collect_rollout_jax` に `shaping_clip: float = 0.0` 引数追加。
    - `step_fn` 内 shaping 算出後に `clip>0` のとき `jnp.clip(shaping, -clip, +clip)` を適用
      (clip<=0 は no-op で既存 mode bit-identical)。vmap in_axes に None 1 本追加。
  - `bot/pipeline/reinforce/case5/training/train_jax.py`
    - `shaping_clip = float(t_cfg.get("shaping_clip", 0.0))` を読み `_run_iter`→rollout に渡す。
    - history row に `shaping_clip` 記録。
  - `bot/pipeline/reinforce/case5/configs/kaggle_jax_train_h7_ratio_clip.yaml` (新規、h4 base + shaping_clip=0.1)
  - `bot/src/gpu/runpod/config/cases.py` に `reinforce_case5_kaggle_jax_train_h7_ratio_clip` stage 追加
- ハイパーパラメータ: `shaping_mode=ratio` / `shaping_coef=1.0` (H4 据え置き) + `shaping_clip: 0.0 → 0.1`。
- データセット / 特徴量変更: なし。

## 実装ステップ (Implementation outline)
1. `rollout_jax.py`: `_rollout_one_env` に `shaping_clip` 引数、`step_fn` で
   `shaping = jnp.where(shaping_clip > 0, jnp.clip(shaping, -shaping_clip, shaping_clip), shaping)`。
2. `collect_rollout_jax` に `shaping_clip` kwarg + vmap `in_axes` を 1 本拡張 (None)。
3. `train_jax.py`: YAML から `shaping_clip` 読込、`_run_iter` シグネチャと rollout 呼び出しに plumb、
   history row 記録。
4. 新 yaml `kaggle_jax_train_h7_ratio_clip.yaml` (h4 全コピー + `shaping_clip: 0.1`)。
5. `cases.py` に stage 追加。
6. ユニットテスト: clip>0 で報酬が [-clip,clip] に収まる / clip=0 で既存 ratio と bit-identical /
   rollout 非発散。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)
- ローカル self-play 300 対戦は行わない (学習中 last-10 win_rate + trend で採否)。
- Kaggle publicScore / skill rating 不使用。
- n<300 で確定判定しない (default ON) → win-rate は inconclusive 固定、trend で傾向判断。
- replay 分析は学習ログ base。
- 例外条件: なし。

### 実施する検証
- ローカル: `dev/test-bot` + `uv run --directory bot pytest tests/unit/pipeline/reinforce/case5 -x`
- smoke: clip 適用後の報酬が [-clip,clip] 内 / clip=0 非破壊をユニットテストで担保。
- リモート: `dev/runpod train <sha> --case reinforce_case5_kaggle_jax_train_h7_ratio_clip --gpu-name "NVIDIA GeForce RTX 3090" --gpu-name "NVIDIA GeForce RTX 4090"` (3090/4090 限定)、~2.5h。
- 評価: 対戦相手 baseline_jax_lite (in-training)、128 ep/iter、主要メトリクス = lite phase last-10 win_rate + trend、
  採否しきい値 = H4 (ratio count, last-10 0.820) と比較し +3pp で clip 採用、悪化/同等なら H4 維持 (clip 不要を確認)。

## リスク / 既知の不確実性
- H4 で value_loss/approx_kl が既に安定 → clip の改善余地が小さく inconclusive/同等が最有力 (事前見込み)。
  本 iter は「ratio 正規化だけで spike 制御は十分、追加 clip は不要」という負の結論の確定が主目的。
- band clip は厳密には PBRS telescoping を崩す (Σ が打ち消されない) が、ratio mode は Φ∈[0,1] で
  境界では ΔΦ が小さく、clip=0.1 は序盤の極端 spike のみを削る軽度の整形。過小なら効果ゼロ、
  過大なら通常 signal も削るため 0.1 を採用。trend/value_loss で副作用を監視。
