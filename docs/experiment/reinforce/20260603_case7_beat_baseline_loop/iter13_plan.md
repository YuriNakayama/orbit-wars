# case7 ループ — iter13 PLAN: ladder で強化した model を lite へ resume

時刻: 2026-06-03 13:05 (cron tick 17)

## ★鍵となる発見 (probe)
iter12 の ladder 学習で強くなった model を **baseline_jax_lite (v1相当) と probe**:
- 結果: outcomes `[-2.0, -2.0, -2.0, +2.0]`、**win_rate 0.25**。
- 以前 (弱い model) は lite で常に -2.0 飽和 → 勾配消失だった。
- **今は 4戦に 1 勝 = reward が飽和していない = lite から学習できる状態になった!**

これがブレークスルーの可能性: ladder で「勝てる相手」を踏み台に強くした結果、
次の階段 (lite=v1相当) で非飽和の勾配が得られるようになった。

## iter13 方針
- iter12 best から **resume**、curriculum `self_snapshot(2) → baseline_jax_lite`。
- horizon=500、ratio/1.0、14 iter。
- lite で 0.25 → さらに上がるか (勾配が効くか) を確認。

## 期待 / 判定
- lite vs win が iter を通して 0.25 → 上昇すれば「梯子を一段登れた」証拠。
- 完走 → 10戦 vs baseline_v1 (lite で強くなれば v1 にも transfer する可能性)。
- これは「弱い相手で踏み台 → 少し強い相手へ」の段階的攻略の実践。

## 留意
- best.pt は win_rate>=best で保存 = lite stage で win 上がった時に保存される
  (self stage の 1.0 で上書きされる懸念 → 必要なら後で best 選択ロジック調整)。
- v1 直接はまだ飽和の可能性 → まず lite を攻略してから v1 を再 probe。
