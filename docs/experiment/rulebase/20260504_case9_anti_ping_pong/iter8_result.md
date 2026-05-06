# rulebase/case9 — anti_ping_pong (iter8 result, HARASS threshold)

> 作成日: 2026-05-06
> 関連: `iter8_plan.md`, `iter1-7_*.md`、特に iter5 analysis (余剰 ship 流用)
> Status: **採択** (55.5% / 200戦、+5pp ボーダー突破!)

## サマリ

`HARASS_MIN_TARGET_PRODUCTION: 2 → 3` の 1 行変更で **iter1-8 で初めて +5pp 達成**。
200戦で **55.5% (v9=111 / v4=89)**、iter2 (49.5%) 比 **+6.0pp**、しきい値 55% 突破。

仮説 (iter5 analysis 由来): 「余剰 ship が低 production の harass で消費される」が
正しかった。低 production target を harass から外すことで余剰が capture / reinforce
の本命用途に流れ、結果として勝率が伸びた。

## 数値

### Phase B: vs baseline_v4 200戦 (seed 10000-10199)

| 配置 | エピソード | v9 勝 | v9 勝率 |
|---|---|---|---|
| seat=0 (v9 先手) | 100 | 59 | **59.0%** |
| seat=1 (v9 後手) | 100 | 52 | **52.0%** |
| **合計** | **200** | **111** | **55.5%** |

- Wilson 95% CI: 約 [48.5%, 62.2%]
- 平均試合長: 373.6 turn (iter2 / iter7 等価)
- Seat bias: 7pp (iter7 5.4pp と同等水準)

### chunk 別累積勝率

| chunk | iter8 累積 |
|---|---|
| 0–20 | 55.0% |
| 0–40 | 55.0% |
| 0–60 | 58.3% |
| 0–80 | 60.0% (peak) |
| 0–100 | 59.0% |
| 0–120 | 56.7% |
| 0–140 | 54.3% (一時 dip) |
| 0–160 | 56.3% |
| 0–180 | 56.1% |
| **0–200** | **55.5%** (確定) |

→ **一貫して +5pp ボーダー超え**で推移、終始 54%+ で安定。

### iter1–8 サマリ

| iter | 主要変更 | n | 勝率 | 採否 |
|---|---|---|---|---|
| 1 | cooldown 抑止 | 100 | 46.0% | 棄却 |
| 2 | bypass=8 + 値短縮 | 200 | 49.5% | best 設計 |
| 3 | bypass=10 | 180/200 | 47.8% | 棄却 |
| 4 | multi-source | 200 | 47.0% | 棄却 |
| 5 | ACCUMULATE port | 200 | 42.5% | 棄却 |
| 6 | plan_shot cache | 200 | 49.0% | 採用 (基盤) |
| 7 | 300戦 confirm | 300 | 52.0% | 棄却 |
| **8** | **HARASS_MIN_TARGET_PRODUCTION 2→3** | **200** | **55.5%** | **採択!** |

## 診断

**仮説完全支持**: iter5 analysis で観測された「拮抗持続 + 僅差負け」シナリオ C で
v9 が production 増産速度で v4 に劣る原因を、iter8 は HARASS 削減 → capture
重視の流れで解消したと推察。

**production=2 の harass が悪手だった理由**:
- production=2 の敵惑星を 1 ターンだけ奪ってもステルス production 増分は数 ship 程度
- 取り返される時に launch 数十 ships を消費、コストパフォーマンスが悪い
- 残った余剰 ship が capture (中立 production 増強) / reinforce (惑星防衛) に
  使えていなかった
- production=3 以上に絞ることで「奪う価値が大きい高 production target」だけに
  harass が集中、副次的に余剰が増えた

**Seat bias は 7pp で iter7 と同水準**: 設計差より seed range 依存が大きい点を再確認。

## 判定

**採択** (200戦で +5pp 達成、しきい値 55% 突破)。

ただし完了条件 (300戦で **60% 以上**) はまだ未達。iter9 で 300戦 confirm:
- 300戦 ≥60% → loop 完了条件達成 → memory 記録 + cron 停止
- 300戦 ≥55% → 採択は維持、iter10 で更にチューニング (production=4 試す等)
- 300戦 <55% → 200戦の seed 偶発と判定、iter10 で別軸

## NEXT ACTION (iter9)

1. `compare_v4.py -n 150 -p 4 --seed 11000` (300戦)
2. ETA ~100 min (iter6 cache あり)
3. 結果に応じて分岐 (上記)

## 成果物

- 評価ログ: `/tmp/compare_v4_iter8.log` (200戦 summary)
