# Reinforce/case5 — support_reward (iter3)

> 作成日: 2026-05-28
> 仮説 ID: H4 (係数 sweep — ratio 文脈に再定義)
> hypotheses.md: docs/experiment/reinforce/20260527_case5_support_reward/hypotheses.md
> 関連: iter2_plan.md / iter2_result.md (H2 ratio: lite last-10 0.763 / trend +0.651, adopted)
> スコープ: ratio mode の shaping_coef を 0.50 → 1.0 に上げ、ratio 信号を強めて last-10 0.763 を超えるか検証

## 仮説 (Hypothesis)
H2 ratio (coef=0.50) が last-10 0.763 と圧勝した。ratio potential は [0,1] 正規化のため
ΔΦ は 1 turn 当たり ≤1 と小さい。**shaping_coef を 1.0 に倍増**すれば ratio 信号が強まり、
収束をさらに加速 / last-10 を押し上げる可能性がある。一方、diff mode では coef=1.0 が
over-shaping (value_loss 悪化) だった前例があるため、ratio でも過剰整形にならないかを同時に見る。

## 既存コードの現状 (from Step 1)
- `rollout_jax.py`: ratio mode は `_shaping_coefs` で `(shaping_coef, shaping_coef)` を返す。
  → shaping_coef を変えるだけで ratio 係数 sweep が可能 (**コード変更不要、config のみ**)。
- iter2 所見: coef=0.50 で last-10 0.763 / trend +0.651 / value_loss 0.005 (極小)。
  value_loss に余裕があるので coef を上げる余地は大きい。

## スコープ (Scope)
- 変更ファイル:
  - `bot/pipeline/reinforce/case5/configs/kaggle_jax_train_h4_ratio_coef1.yaml` (新規、h2_ratio ベースで shaping_coef のみ 1.0)
  - `bot/src/gpu/runpod/config/cases.py` に `reinforce_case5_kaggle_jax_train_h4_ratio_coef1` stage 追加
- ハイパーパラメータ: `shaping_coef: 0.50 → 1.0` (ratio mode、ship/planet 両割合に等加算)
- コード変更: なし (ratio mode 既存、config sweep のみ)。

## 実装ステップ (Implementation outline)
1. `kaggle_jax_train_h2_ratio.yaml` をコピーし `kaggle_jax_train_h4_ratio_coef1.yaml` を作成、shaping_coef=1.0 に。
2. `cases.py` に stage 追加。
3. (コード変更なしのため新規ユニットテスト不要、既存 case5 テストが非破壊を担保)。

## 検証方法 (Validation method)

### スキップする検証 (from hypotheses.md skip list)
- ローカル self-play 300 対戦は行わない (学習中 last-10 win_rate + trend で採否)。
- Kaggle publicScore / skill rating 不使用 (project rule)。
- n<300 で確定判定しない (default ON) → win-rate は inconclusive 固定、trend で傾向判断。
- replay 分析は学習ログ base。
- 例外条件: なし。

### 実施する検証
- ローカル: `dev/test-bot` + `uv run --directory bot pytest tests/pipeline/reinforce/case5 -x` (非破壊確認)。
- smoke: コード変更なしのため既存テストで担保 (config のみ変更)。
- リモート: `dev/runpod train <sha> --case reinforce_case5_kaggle_jax_train_h4_ratio_coef1 --gpu-name "NVIDIA GeForce RTX 3090" --gpu-name "NVIDIA GeForce RTX 4090"` (3090/4090 限定、A100 除外でコスト抑制)、~3h。
- 評価: 対戦相手 baseline_jax_lite (in-training)、128 ep/iter、主要メトリクス = lite phase last-10 win_rate + trend、
  採否しきい値 = H2 (coef=0.50, last-10 0.763) と比較し +3pp で coef=1.0 採用、悪化なら 0.50 維持。

## リスク / 既知の不確実性
- ratio ΔΦ は小さいので coef 倍増でも over-shaping にならない見込みだが、value_loss / approx_kl を監視。
  もし悪化 (value_loss 上昇 or approx_kl > 0.05) なら coef=1.0 は rejected、deepen で coef=0.25 を試す余地。
- 3090/4090 在庫切れ時は launch 失敗 → 時間を置いてリトライ。
