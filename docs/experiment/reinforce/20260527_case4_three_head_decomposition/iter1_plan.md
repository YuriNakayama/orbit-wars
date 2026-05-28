# Reinforce/case4 — 3-head 分解 (iter1) PLAN

> family: reinforce / case: case4 / mode: one-shot (hypotheses.md なし)
> branch: feature/noop-classification-head

## 仮説

reinforce/case3 の action head は 2-head 分解:

```
P(action) = P(Y) · P(X | Y)
  P(Y)    (P+1)-class categorical — target 惑星 + NO_OP を一括
  P(X|Y)  Normal — log1p(ships)
```

これを launch decision を独立させた 3-head 分解に置換する:

```
P(action) = P(Z) · P(Y | Z=1) · P(X | Y, Z=1)
  P(Z)       launch head — sigmoid(launch_logit)   2値 (撃つ/撃たない)
  P(Y|Z=1)   target head — softmax over P planets  (NO_OP slot なし)
  P(X|Y,Z=1) ship head   — Normal(ship_mean, σ)    log1p(ships)
```

数学的には同じ族を表現できるが、「行動の有無 (Z)」と「行動先 (Y)」を明示的に
分離することで PPO の学習信号がクリーンになり、**性能 (勝率) と収束速度
(trend / last-10)** が改善する、という仮説。

## 採否しきい値

case3 (2-head) を同 seed・同レシピ・200 iter で再学習した base に対し、
case4 (3-head) の **last-10 平均勝率が +0.03 以上** なら採用候補。
trend (chunk 単調増加) と best も補助指標として併記。

> 単発 (n=2 run) のため最終判定は inconclusive 寄り。seed variance を考慮し、
> 差が小さい場合は「有意でない」と明記する。

## スコープ

- 変更は **JAX 経路のみ** (`model_jax.py` / `sampling_jax.py` /
  `sampling_eval_jax.py` / `ppo_jax.py`)。学習は JAX で回るため十分。
- backbone / featurizer / 学習レシピ (shaping=0.50 / curriculum sw=5 /
  lr 3e-5→3e-6 / 128 ep/iter / 200 iter) は case3 と完全同一。
- PyTorch 経路 (`model.py` 等) は case3 の 2-head のまま (inference 再ロードは
  follow-up、本実験のスコープ外)。

## 検証方法

1. **smoke** — JAX 6-iter (`kaggle_jax_smoke.yaml`) を CPU で実行し rollout →
   GAE → PPO update の配線と log_prob 整合性を確認 (済: sample.log_prob ==
   eval.log_prob を 1e-4 で確認、6-iter 完走)。
2. **base 再学習** — case3 (2-head) を `kaggle_jax_train.yaml` (200 iter, seed=0)
   で RunPod 再学習。
3. **本命** — case4 (3-head) を同条件 (200 iter, seed=0) で RunPod 学習。
4. **比較** — 両 run の `metrics.json > history[].win_rate` から
   trend (前半→後半 chunk 平均差) / last-10 平均 / best を head-to-head 比較。

> 評価は学習ログの win_rate 曲線ベース (RL の rollout 内 self-play 勝率)。
> Kaggle publicScore は用いない。

## Numbers (記入は result.md)

| metric | case3 (2-head base) | case4 (3-head) | Δ |
|---|---|---|---|
| last-10 平均勝率 | | | |
| trend | | | |
| best | | | |

## Artifacts (予定)

- case3 base: `data/output/models/reinforce/case3_train_jax/runs/<run_id>/`
- case4 3-head: `data/output/models/reinforce/case4_train_jax/runs/<run_id>/`
