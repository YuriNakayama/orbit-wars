# Reinforce/case4 — 3-head 分解 (iter1) RESULT

> 関連: iter1_plan.md
> case4 run_id: 20260527-142530__feature-noop-classification-head__5073ccd__seed0 (3-head)
> case3 run_id: 20260527-143710__feature-noop-classification-head__5073ccd__seed0 (2-head base)
> commit: 5073ccd / GPU: RTX 4090 ×2 / seed: 0 / iterations: 200
> 開始: 2026-05-27 14:25 (case4) / 14:37 (case3) / コスト: ~$2.0 (2 run 合計)

## Summary

仮説は **棄却**。launch decision を独立 head に切り出した 3-head 分解
P(Z)·P(Y|Z)·P(X|Y,Z) は、case3 の 2-head 分解 P(Y)·P(X|Y) に対して
**last-10 平均勝率で −6.2pp 劣化** (0.408 vs 0.470)。学習レシピ・backbone・
seed・SHA を完全に揃えた head-only ablation で、収束曲線のほぼ全 chunk で
2-head が 3-head を上回った。trend (右肩上がりの傾き) は両者ほぼ同等
(+0.150 vs +0.139) なので「3-head が遅れて追いつく」兆候もない。

数学的には同じ族を表現できるが、**明示分離はむしろ最適化を難しくした**と解釈
できる: 2-head は NO_OP も含めた (P+1) softmax 一発で「撃つか・どこへ撃つか」を
同時に competition させるのに対し、3-head は launch (Bernoulli) と target
(softmax) の勾配が分離し、launch head の sigmoid 飽和が target 学習へ波及
しにくくなった可能性。

## 可視化

収束曲線 (10-iter 移動平均) と chunk 平均の比較図:
`data/output/experiment/reinforce_case4_3head_vs_2head.png`

## Numbers

学習ログ (`train.log`) の `iter=N ... win=X` 系列ベース。case4 の pod は
iter ~186 で S3 同期後に終了し train.log は iter 0–185 まで同期 (best.pt は
最終重み)。case3 は iter 0–199 完走。**公平比較のため両者を先頭 186 iter に
truncate** した値を主指標とする。

| metric (先頭 186 iter) | case4 (3-head) | case3 (2-head base) | Δ (3−2) | 採否しきい値 |
|---|---|---|---|---|
| **last-10 平均勝率** | 0.408 | **0.470** | **−0.062** | +0.03 で採用候補 → 未達 |
| best | 0.992 | 1.000 | −0.008 | (early curriculum の fluke、参考外) |
| trend (後半5 − 前半5 chunk) | +0.150 | +0.139 | +0.011 | — |

参考: case3 を 200 iter 完走で見ると last-10=0.487 / trend=+0.145
(case1 AA 300iter 0.501 / Z v2 200iter 0.491 と同水準で、case3 base 自体は健全)。

### chunk 平均勝率 (10 分割、先頭 186 iter)

```
chunk         1     2     3     4     5     6     7     8     9    10
case4 3-head 0.300 0.119 0.163 0.192 0.227 0.281 0.315 0.373 0.379 0.405
case3 2-head 0.354 0.170 0.264 0.314 0.359 0.379 0.399 0.455 0.464 0.460
diff (3−2)  -0.054 -0.051 -0.101 -0.122 -0.132 -0.098 -0.084 -0.082 -0.085 -0.055
```

warmup の dip (chunk 2) 以降、全 chunk で 2-head がリード。差は中盤
(chunk 3–5) で最大 −0.10〜−0.13、終盤でも −0.05〜−0.09 で縮まらない。

## Diagnosis

head 構造以外は完全同一なので差分は分解構造に帰属できるが、補助指標を読むと
「3-head 化そのものが原理的に劣る」というより **2-head 用にチューニングした
`no_op_bias` / `entropy_coef` を 3-head にそのまま流用したミスマッチ** が支配的
と判断する。

### 補助指標 (train.log)

| 区間 | 3-head win / ploss / approx_kl | 2-head win / ploss / approx_kl |
|---|---|---|
| early 0–30 | 0.231 / **+0.0047** / **0.0222** | 0.281 / −0.0005 / 0.0019 |
| mid 60–120 | 0.241 / +0.0005 / 0.0048 | 0.359 / −0.0003 / 0.0042 |
| late 156–185 | 0.394 / −0.0004 / 0.0039 | 0.455 / −0.0004 / 0.0033 |

3-head は **early の approx_kl が 2-head の約12倍 (0.022 vs 0.0019)** かつ
policy_loss が正 — 序盤に方策が大きく暴れたのに勝率に結びついていない。

### 主因 1: `no_op_bias=8.0` の意味が 2 head 間でズレた (最有力)

- 2-head: `(P+1)` softmax の NO_OP slot を −8 → **P 個の発射先との相対競合**の
  中の 1 票。有望 target が複数あれば自然に発射が選ばれ、P が大きいほど NO_OP
  確率は薄まる。
- 3-head: launch logit を単独で −8 → `sigmoid(−8) ≈ 0.0003`。**target 数と
  無関係に P(撃つ) が固定的に潰れる**。探索初期に「ほぼ何も撃たない」状態へ強く
  張り付き、PPO がそれを引き上げようとして early kl=0.022 の暴れを生んだと見る。
  同じ `8.0` でも 2 head で意味が全く違うのに値を流用したのが効いた。

### 主因 2: 勾配経路の分離で target head の信号が枯渇

`logP = logP(Z) + Z·(logP(Y|Z)+logP(X))` のため **Z=0 サンプルは target/ship
head に勾配を流さない**。主因 1 で序盤の大半が Z=0 になり、target head の有効
データが激減。2-head は NO_OP 込みで全 source の softmax が毎回更新されるので
target 表現の学習効率で差がつく。mid で勝率差が最大 −0.12 に開くのと整合。

### 主因 3: credit assignment / 探索の二重減衰

発射が悪手のとき 2-head は「その target logit を下げる」1 経路。3-head は
launch を下げるか target を変えるかに advantage が按分され帰責が曖昧。entropy も
`H(Z)+P(Z=1)·(H(Y)+H(X))` の積構造で、launch 飽和時に target の探索 bonus まで
`P(Z=1)≈0` で減衰し、探索が二重に細る。

> best=0.99/1.0 は両者とも curriculum 序盤 (vs noop opponent) の勝利体験に
> 由来する fluke で、head 構造の優劣を測る指標にならない。last-10 が本指標。

## Decision

- **採否: rejected** (last-10 −6.2pp、しきい値 +3pp に対し逆方向)
- n=1 run/構成だが seed/recipe/SHA を揃えた直接 ablation で差が大きく方向も
  明確なため、inconclusive ではなく rejected と判定。
- **次の一手**:
  - case4 (3-head) は採用しない。canonical の reinforce head は case3 系
    (2-head) を維持。
  - 追検証するなら主因 1–3 への直接対処: (a) launch bias を −8 でなく
    −2〜−3 に緩め初期 P(撃つ)≈10% を狙う、(b) launch head に専用 entropy 係数を
    与え早期飽和を抑える、(c) target を NO_OP 込み (P+1) のまま launch head を
    **補助** (auxiliary) として足す hybrid。ただし 2-head が既に健全に右肩上がり
    (last-10 0.49) なので ROI は低い。
  - `dev/runpod promote` / Kaggle submit は本 case では不要 (要承認、対象外)。

## Artifacts

- case4 (3-head) model: `data/output/models/reinforce/case4_kaggle_jax_train/runs/20260527-142530__feature-noop-classification-head__5073ccd__seed0/best.pt`
- case4 train.log: 同ディレクトリ `train.log` (iter 0–185)
- case3 (2-head base) model: `data/output/models/reinforce/case3_kaggle_jax_train/runs/20260527-143710__feature-noop-classification-head__5073ccd__seed0/best.pt`
- case3 metrics.json (完走): 同ディレクトリ `metrics.json` (iter 0–199)
- 実装: commit 5073ccd (`bot/pipeline/reinforce/case4/policy/{model_jax,sampling_jax,sampling_eval_jax}.py`, `training/ppo_jax.py`)

## 既知の運用上の注記

- case4 の pod は SECURE/RTX4090 で iter ~186 時点の S3 同期後に終了し、
  `train.py` 末尾の `metrics.json` 書き出しに到達しなかった (best.pt は
  S3 layer-1 safeguard で各 best 更新時に退避済み)。判定は train.log で十分
  完結したが、JAX train_jax は **iter 後半でも train.log を定期 flush + 終了時
  に metrics.json を確実に S3 退避する** 仕組みが弱い (memory
  [[project_runpod_best_pt_lost]] と同根)。次回の長期 run 前に train.log の
  逐次 S3 同期化を検討。
