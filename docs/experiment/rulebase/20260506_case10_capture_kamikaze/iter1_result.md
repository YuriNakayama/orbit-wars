# rulebase/case10 — capture_kamikaze (iter1 result)

> 作成日: 2026-05-06
> 関連: `iter1_plan.md`, `iter2_plan.md`
> Status: **棄却** (45.5% / 200戦、case4 default より -4pp 悪化)

## サマリ

case4 base + 5 定数の同時変更 (capture 強化 + sniper/kamikaze 多用) は **iter2 の case4 同等水準を大きく下回り棄却**。
200戦で **45.5% (v10=91 / v4=109)**。Wilson 95% CI [38.7%, 52.5%]、+5pp ボーダー (55%) には完全に未達。

iter1 の 5 定数変更は case4 default のチューニング結果に逆行する方向だったと判断。

## 数値

### Phase B: vs baseline_v4 200戦 (seed 12000-12199)

| 配置 | エピソード | v10 勝 | v10 勝率 |
|---|---|---|---|
| seat=0 (v10 先手) | 100 | 49 | **49.0%** |
| seat=1 (v10 後手) | 100 | 42 | **42.0%** |
| **合計** | **200** | **91** | **45.5%** |

- 平均試合長: 378.9 turn
- Seat bias: 7pp (v9 系 iter と同水準)

### chunk 別累積勝率

| chunk | 累積勝率 |
|---|---|
| 0–20 | 60.0% (peak) |
| 0–40 | 50.0% |
| 0–60 | 53.3% |
| 0–80 | 50.0% |
| 0–100 | 49.0% |
| 0–120 | 50.0% |
| 0–140 | 50.7% |
| 0–160 | 48.1% |
| 0–180 | 46.1% |
| **0–200** | **45.5%** |

→ 序盤 0-20 戦の peak 60% は seed 偶発、以降一貫して 50% 帯から下方圧力。

### 変更内容 (recap)

| 定数 | case4 default | iter1 | 変更方向 |
|---|---|---|---|
| `STATIC_NEUTRAL_VALUE_MULT` | 1.4 | 1.6 | capture 強化 |
| `EARLY_NEUTRAL_VALUE_MULT` | 1.2 | 1.4 | 序盤 capture 強化 |
| `SNIPE_VALUE_MULT` | 1.12 | 1.30 | snipe 多用 |
| `HARASS_MIN_SRC_RESERVE` | 10 | 6 | kamikaze 多用 |
| `HARASS_PRODUCTION_STEAL_TURNS` | 5 | 8 | harass 価値上昇 |

## 診断

仮説 (capture 強化 + 攻撃寄り設定で +5pp) は **不成立**。理由 (推察):

1. **5 定数同時変更で因果が混ざる**: 個別効果が打ち消し合った可能性
2. **case4 default はすでに tuning 済**: 上向き方向の調整は逆効果になりやすい
3. **HARASS_MIN_SRC_RESERVE=6 が裏目**: kamikaze 多用で **自惑星防衛が手薄**、後手 (seat=1) で 42% に大きく低下したのはこれが原因の可能性
4. **memory の警告**: `project_heuristic_search_saturation` (heuristic 系 53% 飽和)、`project_case9_anti_ping_pong_2026_05_06` (cooldown 系小修正は不可) と整合

## 判定

**棄却**。iter2 では `iter2_plan.md` の **「iter1 棄却 → 逆方向 (defense 寄り)」** ブランチへ:
- `STATIC_NEUTRAL_VALUE_MULT: 1.6 → 1.2` (default 1.4 より低く、capture を意図的に弱める)
- `HARASS_MIN_SRC_RESERVE: 6 → 14` (kamikaze 抑制、defense 重視)
- 残り 3 定数 (`EARLY_NEUTRAL_VALUE_MULT`, `SNIPE_VALUE_MULT`, `HARASS_PRODUCTION_STEAL_TURNS`) は default に戻す

「逆方向で勝率改善するなら iter1 の方向が逆効果と確定、改善しないなら heuristic 系探索の 53% 壁が再確認される」という判定材料。

## 連敗カウント

iter1 で 1 連続棄却。
- 5 iter 連続棄却で loop 終了 (cron prompt のガード)
- 3 連続で memory 11 連敗パターン類似なら早期終了判断
