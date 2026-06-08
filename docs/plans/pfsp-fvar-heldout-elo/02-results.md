# 結果: H5 f_var + held-out + Elo

run_id: 20260608-072926__feature-pfsp-fvar-heldout-elo__0b06b96__seed0
GPU: RTX 4090 / config: h5_fvar_heldout.yaml (80 iter)

## 検証する問い
1. f_var で match 勝率が ~0.5 帯に保たれるか (matchmaking が機能するか)。
2. **held-out 勝率 / Elo が単調 ↑** するか (= 実力が本当に上がるか)。
3. entropy が崩壊しないか (健全性)。
4. held-out 勝率が H1-H4 の ~0.3 天井を超えるか。

## 結果
<!-- TODO: GPU run 後に記入 -->

| 指標 | 値 |
|---|---|
| match 勝率 (per-iter, 平均) | <!-- ~0.5 期待 --> |
| held-out 勝率 (iter0 / 中盤 / 最終) | <!-- --> |
| agent_elo (iter0 / 最終) | <!-- --> |
| entropy (最小値) | <!-- collapse=<10 --> |
| 外部 paired (vs baseline_v8, n=30) | <!-- --> |

## 判定
<!-- held-out↑ かつ entropy健全 → matchmaking 有効。天井超えるか。 -->

## コスト
<!-- --> 
