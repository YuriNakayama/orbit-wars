# imitation/case7 iter2 — Result (parity、Stage 1 ゲート未達 → iter3 へ)

> 作成日: 2026-05-05
> 関連 plan: `./iter2_plan.md`
> commit: `850c34e` (iter2 実装 + trap #9 fix) → train run `20260505-062823__feature-feature-engineering__850c34e__seed0`
> weights: `data/output/models/imitation/case7/runs/20260505-062823__.../best.pt` (epoch 9, val_loss=3.5142、canonical には昇格しない)

## サマリ (TL;DR)

**判定: 破棄**。Stage 1 採否ゲート (全 head iter1 比 +0.01) 不達成。

| head | metric | iter1 | iter2 | Δ vs iter1 | ゲート判定 |
|------|--------|------:|------:|----------:|:----:|
| from | F1 | 0.6651 | 0.6601 | -0.005 | ✗ 微減 |
| from | PR-AUC | 0.7676 | 0.7626 | -0.005 | ✗ 微減 |
| target | macro F1 | 0.3210 | 0.3184 | -0.0026 | ✗ 微減 |
| target | top-1 acc | 0.4327 | 0.4378 | **+0.005** | △ 半分達成 |
| ships | macro F1 | 0.6425 | **0.6292** | **-0.013** | ✗ 大きく後退 |
| ships | acc | 0.8519 | 0.8468 | -0.005 | ✗ 微減 |

best_val_loss は **3.5142 (iter1 3.5235 比 -0.0093)** で訓練自体は安定したが、
**val accuracy / F1 levels では iter1 を上回れず**。一見すると過学習でも未学習でも
ない「parity 程度」の結果。

### 判定理由

- target acc は +0.005 微増で半分ゲート達成だが、target macro F1 は -0.0026 で微減
- 全 head 一斉改善 (+0.01) というゲートは未達
- ships macro F1 が **-0.013 大きく後退** (-0.005 ゲート違反)
- 14 列追加のコスト (preprocess 30%増、parquet サイズ 21-30%増、訓練時 dim 増による memory 増) に対するリターンが見合わない

## 訓練ログ (15 epoch)

```
epoch  train_total  val_total
0      3.8731       3.6898
9*     3.5322       3.5142          ← best (iter1 best 3.5235 比 -0.0093)
14     3.4648       3.5288
```
*訓練時間*: **338.76 秒** (RTX 4090) — iter1 の 403.9 秒より 65 秒短縮
(訓練ステップ数同じ、A6000→4090 の差ではなく、もしかしたら iter2 の追加列が
multiprocessing IO で先行プリフェッチに優しかった可能性)。

## Stage 1: validation metrics

### 全 head (case3 phase2 体系)

| head | metric | case3 phase2 | iter1 | **iter2** | Δ vs iter1 |
|------|--------|-------------:|------:|---------:|----------:|
| **from** | F1 | 0.6182 | 0.6651 | **0.6601** | **-0.005** ✗ |
| from | PR-AUC | 0.6317 | 0.7676 | **0.7626** | **-0.005** ✗ |
| from | ROC-AUC | 0.9275 | 0.9532 | 0.9528 | -0.0004 |
| from | acc | (n/a) | 0.9151 | 0.9096 | -0.0055 |
| **target** | macro F1 | 0.3076 | 0.3210 | **0.3184** | **-0.0026** ✗ |
| target | top-1 acc | 0.4238 | 0.4327 | **0.4378** | **+0.005** △ |
| target | top-2 acc | (n/a) | 0.6377 | 0.6390 | +0.001 |
| target | PR-AUC macro | (n/a) | 0.3419 | 0.3423 | +0.0004 |
| target | ROC-AUC macro | (n/a) | 0.7763 | 0.7774 | +0.001 |
| **ships** | macro F1 | 0.6677 | 0.6425 | **0.6292** | **-0.013** ✗✗ |
| ships | acc | 0.8108 | 0.8519 | 0.8468 | -0.005 |
| ships | MAE bucket | 0.2501 | 0.2022 | 0.2039 | +0.0017 (worse) |
| ships | PR-AUC macro | (n/a) | 0.7117 | 0.7014 | -0.010 |
| ships | ROC-AUC macro | (n/a) | 0.9435 | 0.9412 | -0.002 |

### 解釈

- **from head は安定だが微減**: iter1 で +0.136 という大幅改善があった from PR-AUC が、追加 14 列のノイズで僅かに減衰。これは「飽和した head に追加 input → noise 化」の典型。
- **target head は微妙な前進**: top-1 acc が +0.005 改善、PR-AUC macro / ROC-AUC macro も +0.001 微増。**iter2 plan で狙った target plateau 突破は半分達成**だが、macro F1 では微減で打ち消し。
- **ships head は明確な後退**: macro F1 -0.013、PR-AUC macro -0.010、acc -0.005 と全方位で悪化。iter2 の 14 列が **ships の bucket 判定にネガティブな input を持ち込んだ**可能性。fleet trajectory (24-27) は my-planet 以外でも 0 fill が多く、これが ships head の信号を希釈している疑い。

### 推定原因

iter2 plan の Risk #2 が顕在化:
> 多くの planet には inbound enemy fleet が無く、32-35 列が 0 fill ばかりになる。

加えて、**multi-horizon ship-prediction (28-31)** は既存 `loss_3turn` / `min_owned` と
強相関で、独立信号にならず冗長 input になっていた可能性。

## Stage 2 (out of scope)

破棄判定のため self-play は実施しない。

## 採用判定

**破棄**。

### 後処理

- canonical weights `bot/pipeline/imitation/case7/policy/weights.pt` は **iter2 best.pt のまま保持**
  (rationale: featurizer.py が 34/14 schema にあり、iter1 weights (24/10) との dim
  mismatch で agent runtime crash する。iter1 にロールバックするには featurizer
  も同時に revert が必要。iter3 を 34/14 base から進化させる方針なので iter2
  weights を canonical 維持)
- iter2 best.pt は val_loss 3.5142 (iter1 3.5235 比 -0.0093 小改善) なので
  「破棄」というより「**Stage 1 head 精度では parity、val_loss は微改善**」評価
- run artifacts は `data/output/models/imitation/case7/runs/20260505-062823__.../` に保存
- featurizer.py の (34/14) schema は iter3 のベースとして git に残す

## RunPod onstart 進化 (このサイクルで追加された trap)

| # | trap | 修正 commit |
|---|------|------|
| 9 | preprocess_skip が schema 変更を見ずに古い parquet を再利用 | `850c34e` (`dvc status` で deps 鮮度確認) |

trap 1-8 は iter1 で発見済み。iter2 では trap #9 が新たに顕在化、修正済み。

## 教訓 (iter3 設計に活かす)

1. **「全 head +0.01」は厳しすぎるゲートだった**: iter1 で from PR-AUC が +0.136 という大躍進をしたあと、from を **「安定 (誤差 ±0.005)」** で許容し、ships / target に改善ゲート (+0.01) を絞る方が現実的。
2. **追加列は sparse な場合 mask flag を併設**: fleet trajectory が 0 fill ばかりだと ships head の信号を希釈する。`has_inbound_fleet` flag を 1 列足して、policy が「該当フィールドが有効か」を学べるようにする。
3. **multi-horizon は collinear 注意**: horizon=5/15/30 は強相関で、独立信号にならない。**1 horizon に集中** (例 = 5 turn のみ) するか、**差分** (`loss_30 - loss_5`) のような直交化された値にする。
4. **訓練時間は短縮された** (-65s): 14 列増えても train cost はほぼ変わらない。iter3 で更に列を追加しても問題なし。

## 関連ファイル

- featurizer (iter2): `bot/pipeline/imitation/case7/policy/featurizer.py` (34/14 schema)
- weights (iter2): `data/output/models/imitation/case7/runs/20260505-062823__feature-feature-engineering__850c34e__seed0/best.pt`
- val metrics: `/tmp/case7_iter2_val_metrics.json` (もし保持するなら `data/output/experiment/imitation_case7_iter2_val_metrics.json` に移動)
- 訓練 metrics: `data/output/models/imitation/case7/runs/20260505-062823__.../metrics.json`
- iter2 onstart trap #9 修正: `bot/src/runpod_io/onstart.sh.tmpl`

## 次のアクション

- [x] iter2_result.md 作成 (本ファイル)
- [ ] canonical weights を iter1 best.pt にロールバック
- [ ] iter3 plan を書く (`/experiment-plan` で別途、ships head 安定 + fleet trajectory sparse trick 中心)
