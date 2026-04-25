# iter9 勝敗ケース分析と非決定性 finding

## 要約

当初の Phase 2 iter9 評価は 5/100 win と記録されていたが、**同じ weights (pipeline/imitation/case1/policy/weights_iter9.pt) の再評価で 600 戦連続で 0 勝**となり、Phase 2 の "5/100 breakthrough" は統計的に有効でないことが判明した。

### 再評価結果 (同一 weights)

| バッチ | wins / games | source |
|---|---|---|
| b9dplo38t (save_replay) | 0 / 200 | 2026-04-21 再試行 |
| bnu9p7ytn | 0 / 100 | 2026-04-22 再試行 |
| bfbs2285k | 0 / 300 | Phase 3-C baseline 計測 |
| **合計** | **0 / 600** | Wilson 95% CI [0.00, 0.63%] |

### 過去 "win" ファイル (iter9_win_seed19.json) の実態

`dev/replay_one_match.py --seed 19 --label win` で保存されたファイル (3.4MB、全 steps 保存) をデコードした結果:

| フィールド | 値 |
|---|---|
| winner | **baseline_v1** |
| final_rewards | [-1, +1] (il_v1 敗北) |
| turns_played | 171 |

当時 "winner idx = 0" と表示された出力は、**ラベル付けは win だが実行時 reward は loss**。保存時メタデータは `winner_idx` キーで保存されていたが、このファイルの `winner` は `baseline_v1`。つまり **Phase 2 win seed も実体は負け試合**だった疑いが強く、5/100 自体が recorder/runner のカウントタイミング問題か非決定性の偶然ブレの可能性がある。

## なぜ再現できないか (非決定性)

同一 seed / 同一 weights / 同一 git_sha でも結果が変わる。kaggle_environments の内部 step で RNG が副作用として消費される箇所 (fleet tie-breaking, comet 配置 jitter 等) が複数ワーカー並列実行下で非決定化している可能性が高い。

**運用指針:**
- 100 戦以下の評価は信号化できない (CI 上限が base 差分を飲み込む)
- **300 戦 + Wilson 95% CI** を imitation/case1 の標準評価とする
- `pipeline/imitation/case1/evaluation/eval_vs_baseline.py` で `win_rate_ci95_lo/hi` を出力する形に統一済

---

## 敗北ケース分析 (seed=0)

詳細は `loss_seed0_analysis.md` 参照。代表所見のみ:

| 指標 | 値 |
|---|---|
| turns | 126 |
| final score | il=0, base=2854 |
| il_v1 0-action turn | 46/126 (36.5%) |
| 崩壊タイミング | turn 80 (ships 200→123) → turn 100 (planets 5→0) |

中盤 (turn 30-50) の拡張ペース遅れ (base_planets 5→9) → 戦線崩壊 → 全惑星喪失という定型パターン。

---

## 擬似「勝利」ケース分析 (seed=19, 実態は敗北)

当初 win 判定されていた seed=19 の全 steps をトレースした結果:

### 序盤 (turn 5-40) — 一時的に優勢

| turn | il_ships | base_ships | il_p | base_p | il_f | base_f |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 3 | 12 | 1 | 1 | 4 | 1 |
| 10 | 3 | 3 | 1 | 1 | 8 | 3 |
| 20 | 15 | 17 | 2 | 2 | 16 | 6 |
| 40 | 104 | 46 | 6 | 6 | 24 | 16 |
| 60 | 386 | 143 | 11 | 12 | 25 | 38 |

- turn 40 で il_v1 が ships 104 vs 46 と倍近いリード
- turn 60 では ships 386 vs 143 で圧倒的優勢 (**seed=0 敗北ケースとは全く違う序中盤**)
- planets も 11 vs 12 でほぼ互角

### 中盤崩壊 (turn 60-100)

| turn | il_ships | base_ships | il_p | base_p | il_f | base_f |
|---|---:|---:|---:|---:|---:|---:|
| 60 | 386 | 143 | 11 | 12 | 25 | 38 |
| 80 | 555 | 576 | 8 | 22 | 24 | 39 |
| 100 | 439 | 795 | 5 | 23 | 42 | 73 |

- **turn 60 → 80 で planets 11 → 8 (3 落とす)**、同時に base_planets 12 → 22 に倍増
- ships は turn 80 で逆転 (555 vs 576)、turn 100 で完全に離される (439 vs 795)
- **惑星確保の戦略的失敗が ships リードを無駄にしている**

### 終盤 (turn 120-171)

| turn | il_ships | base_ships | il_p | base_p |
|---|---:|---:|---:|---:|
| 120 | 116 | 670 | 1 | 27 |
| 140 | 0 | 2816 | 0 | 28 |
| 171 | 0 | 5882 | 0 | 32 |

turn 140 で il_v1 全惑星喪失 → そのまま終局。

### アクション分布

| turn 内 actions | il_v1 | baseline_v1 |
|---|---:|---:|
| 0 | 58 (33.9%) | 62 (36.3%) |
| 1 | 52 | 24 |
| 2 | 34 | 13 |
| 3 | 20 | 12 |
| 4 | 7 | 11 |
| 5+ | 0 | 49 (28.7%) |
| max | 4 | **28** |

- il_v1 の 0-action turn 率 (33.9%) は seed=0 (36.5%) とほぼ同等
- il_v1 は 4 fire/turn で頭打ち (max_fire_count=4 設定の結果)
- baseline_v1 は 28 fire/turn を打つ局面があり、一度に大規模な反攻をしてくる

---

## 敗北 (seed=0) vs 擬似勝利 (seed=19) の共通パターン

| 指標 | seed=0 (敗北) | seed=19 (元 "win" 実態負け) |
|---|---|---|
| 0-action turn 率 | 36.5% | 33.9% |
| il_v1 serialize 艦数 pickup | turn 20 時点で 42 | turn 40 時点で 104 |
| 相手が伸び始める turn | 30-50 | 60-80 |
| il_v1 惑星喪失開始 turn | 70 | 60-80 |
| 全惑星喪失 turn | 100 | 140 |
| 最終 turn | 126 | 171 |

**結論:** どちらも「序盤〜中盤優勢 → 惑星拡張ペースで徐々に遅れ → 中盤以降 planets が減り続ける → 終盤に ships が 0 になり終局」という同型崩壊。差異は崩壊速度 (seed=19 の方がゆっくり)。

## Phase 3 への示唆

1. **勝利ケースが観測できない** ため、val metric 中心の改善では勝率にリンクしない。惑星確保 (NEAREST_NEUTRAL / HIGHEST_PROD_NEUTRAL) の行動学習が最優先。
2. **max_fire_count=4 がボトルネック** の可能性。baseline は 1 turn 28 fire を繰り出すが il_v1 は最大 4 で頭打ち。inference config の見直し余地。
3. **36% 前後の 0-action 率は勝ち負けとは無関係のハード上限**として存在。from_threshold / max_fire_count の tuning だけでは解消しない、行動サンプリング自体の課題。
4. **300 戦 + CI を基盤にしないと、施策の有効性が判定できない**。iter11 以降はこの評価軸で判断する。
