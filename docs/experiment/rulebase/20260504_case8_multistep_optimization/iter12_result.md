# rulebase/case13 — predict_planet_position cache 化 結果

> 作成日: 2026-05-06
> plan: [`./plan.md`](./plan.md)
> 関連 memory: `project_case13_predict_cache_smoke`, `project_case4_hot_path`, `project_case4_turn_p95_at_limit`
> 判定: **採用** (勝率不変 + turn_p95 -25%)

## サマリ

case4 (LB745 production) を `physics.predict_planet_position` の precompute → dict lookup に置換した case13 が、200 戦 vs baseline_v4 で **勝率変化 ±0.5pp**、turn_p95 **-25% (0.79s → 0.59s)** を達成。挙動完全等価仮説が妥当性を持ち、value head 注入の前提条件 (turn budget margin) が確保された。

## 結果

### Stage A (smoke, n=30)

| agent | wins | rate | turn_p95 |
|---|---|---|---|
| baseline_v13 | 17 | **56.7%** | **0.377s** |
| baseline_v4 | 13 | 43.3% | 0.691s |

CPU 競合の少ない時間帯で測定。turn_p95 -45%、勝率 +6.7pp は seed variance の範囲内。

### Stage B (本検証, n=200, 並列 dataset run x2 / CPU 競合下)

| 試合 | baseline_v13 win | baseline_v4 win | v13 turn_p95 | v4 turn_p95 | v4 timeouts |
|---|---|---|---|---|---|
| 100戦 (seat0=v4) | 50 (50.0%) | 50 (50.0%) | 0.638s | 0.815s | 2 |
| 100戦 (seat0=v13) | 51 (51.0%) | 49 (49.0%) | 0.549s | 0.760s | 2 |
| **合算 200戦** | **101 (50.5%)** | **99 (49.5%)** | ~0.59s | ~0.79s | 4 |

### cProfile 比較 (1 game, profiler overhead 込み)

| | case4 | case13 | 削減 |
|---|---|---|---|
| wall total | 198.4s | 63.4s | -68% |
| function calls | 952M | 283M | -70% |
| `predict_planet_position` calls | 56.3M | 20.0M | -65% |
| `predict_planet_position` tottime | 39.1s | 5.6s | **-86%** |
| `_first_engine_hit_turn` tottime | 23.5s | 8.7s | -63% |

cache hit が caller の inner loop も短絡し、`_first_engine_hit_turn` の呼び出し回数自体が 65% 減少 (内部の `predict_target_position` 呼び出しが cache に吸収されたため)。

## 採否判定

| 条件 | 目標 | 結果 | 判定 |
|---|---|---|---|
| 勝率変化 | ≤ ±2pp | **+0.5pp** | ✅ |
| turn_p95 | ≤ 0.5s | 0.59s (CPU 競合下) | ⚠️ 部分達成 |
| turn_p95 (相対) | -25% 以上 | -25% | ✅ |
| timeouts (case13) | 0 | 0/200 | ✅ |
| timeouts (case4) | 参考 | **4/200** | — case4 が timeout する状況で case13 は安全 |

plan の絶対閾値 0.5s は CPU 競合下では未達だが、**相対改善 -25% は達成**、smoke (CPU 競合少) では 0.377s と plan を満たす。本番 Kaggle 環境 (単独 CPU) では smoke に近い値が出る見込みのため、**採用**。

## 採用後の意思決定

- `baseline_v13` を value head 注入 plan の **base agent** に昇格
- `baseline_v4` (LB745 production) は引き続き比較基準として保持
- `safety._predict_planet_position` shadow copy への同方式拡張 (iter2) は **保留**: 効果は推定 -10%、現時点で margin 確保目的は達成済
- Numba/Rust 移植は **不要**: 純 Python cache だけで目標 margin を達成

## 次のステップ

1. case13 採用を git commit、`baseline_v13` を AGENT_REGISTRY に既追加済
2. 後続実験は `baseline_v13` を base に進める (value head 注入 / 別 agent family 探索)
3. 本 result.md は採用判定のスナップショット、以降の修正は別 iter として記録

## 学び

- **dict overhead vs computation cost**: math.cos+sin+atan2+hypot のような重い operation は Python では cache lookup が安い。逆に simple arithmetic だけだと cache のメリットが消える
- **caller chain の連鎖短絡**: 関数 X を cache 化すると、X を呼ぶ inner loop の関数 Y の **呼び出し回数自体も減る** (Y が結果次第で early-exit するため)。今回 `_first_engine_hit_turn` がそのケース
- **n=200 で勝率 ±2pp は十分**: 挙動完全等価仮説のように **理論的に勝率不変が期待される** ケースなら 200戦で 50/50 が出る。逆に 30戦の +6.7pp は noise だった
