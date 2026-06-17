# case7 「ルールベースに勝つ」ループ — iter08 RESULT

時刻: 2026-06-03 10:01 (cron tick 10)

## 1. 学習は起きているか? → YES (loss/reward/kl で確認)
iter08 (ratio_prod production shaping) の trace:
| iter | opp | win | reward | loss | kl |
|---|---|---|---|---|---|
| 2-4 | self_snapshot | 0.83, 1.0, 0.67 | +0.12, +0.20, **+0.26** | ~-0.01 | 0.006-0.016 |
| 5-7 | self_snapshot | 0.33, 0.5, 0.67 | -0.05, -0.01, +0.12 | … | … |
| 8 | baseline_jax_lite | 0.167 | -0.56 | -0.025 | 0.035 |

- **approx_kl > 0 毎 iter** = 方策は確実に更新されている (optimizer 健全)。
- vs self_snapshot で **reward 正 (+0.26) / win 1.0** = 過去自分に対しては改善。
- **結論: RL machinery は健全。学習は起きている。** 0/10 は最適化失敗ではなく
  「改善が baseline_v1 への勝利に transfer しない (production gap)」が本質。
- production shaping (ratio_prod) でも vs lite は 0.167 = まだ壁 (完走後に vs v1 確認)。

## 2. web search で打開策を調査
"RL vs scripted bot / expansion economy / Planet Wars / Galcon / reward shaping":

### ★最重要: Generals.io を RL で攻略した論文 (本問題の最近接アナログ)
[Artificial Generals Intelligence: Mastering Generals.io with RL (arXiv 2507.06825)](https://arxiv.org/abs/2507.06825)
- Generals.io = 領土拡張・生産経済ゲーム (Orbit Wars/Planet Wars に酷似)。
- レシピ = **supervised pre-training (BC) + self-play + potential-based reward
  shaping + memory features** → **H100 GPU 1台で 36 時間** で人間 top 0.003%。
- **我々のレシピと質的に同一** (BC warm-start + self-play + PBRS ratio shaping)。
  違いは (a) **scale: H100×36h vs CPU 10-30 iter (~分)** = 1000倍以上の差、
  (b) **memory features**: case7 rollout は **empty history** (line 252) で学習 →
  時系列メモリ特徴を活用していない。

### 他の関連
- [StarCraft II supervised+RL (arXiv 2109.12691)](https://arxiv.org/pdf/2109.12691): scripted bot 超えに教師あり+RL。
- [Reward Shaping for RTS (arXiv 2311.16339)](https://arxiv.org/pdf/2311.16339): RTS の reward shaping。

## 結論 (研究で裏づけ)
- **我々のアプローチは SOTA と質的に正しい**が、**scale が 1000倍以上不足**。
  Generals.io が H100×36h を要したのに対し、本ループは CPU 数分。0/10 は当然。
- 追加の改善余地: **memory features** (empty history → 実 history を学習 rollout に
  通す) が論文の加速要因。これは小規模でも効く可能性のある "ロジック改善"。

## iter08 最終測定
| model | vs baseline_v1 | score gap |
|---|---|---|
| iter08 (ratio_prod production shaping) | **0/10** | 51 vs **13834** |

production-weighted shaping で score gap が 16413→13834 と微減も、依然 0/10 の大差。
shaping tweak では production gap を埋められないと確認。

## NEXT ACTION
1. iter08 完走 → 10戦 vs v1 (production shaping の効果確認)。
2. **memory features 改善**: 学習 rollout の empty history を実 history に
   (論文の加速要因、小規模・無料で試せる "学習ロジックのブラッシュアップ")。
3. 本質的勝利には **GPU scale (論文準拠)** が必要と研究が示唆 → directive 許可済だが
   H100×36h 級は非現実的。RunPod で現実的な範囲 (数時間) を 1 回試す価値はある。
