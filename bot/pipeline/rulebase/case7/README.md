# case7 — baseline_v7 (case6 + multi-turn accumulate)

case6 (baseline_v6 = case4 + STAY judge) のフルコピー上に、
**多ターン蓄積からの遠距離単発攻撃 mission** を追加した派生 case。

## 採用戦略

case6 の `STAY_BURST` (1ターン arbitrage) を維持しつつ、新たに
**ACCUMULATE** mission を `baseline/missions/stay.py` に追加。

### 仮説

`fleet_speed = 1 + (max-1) × (log(ships)/log(1000))^1.5` のため
ship 数が多いほど移動速度が上がる。case6 の `STAY_BURST` は
1ターン待ちの局所判断に留まるが、ACCUMULATE は

- 敵脅威スコアが低い友軍 source で
- 「目標惑星の捕獲必要量 + safety + fleet_speed knee」までの ships が
  揃うまで複数ターン蓄積し
- 揃った時点で遠距離 (ETA >= ACCUMULATE_MIN_TARGET_TURNS) の
  友軍 / 敵惑星に単発攻撃する

mission を新設する。これにより case6 の局所最適 (cap=3 で plateau,
300戦 vs v5 で 59.7%) を超える fleet 形成と命中率を狙う。

詳細: `docs/experiment/rulebase/20260504_case7_accumulate_burst/plan.md`

## 成績

- vs baseline_v6 (Stage 0 smoke 100 戦): TBD (`evaluation/compare_v6.py` で測定)

## 構造

```
case7/
├── main.py
├── baseline/
│   ├── agent.py
│   ├── core/                       # case6 と同一 + ACCUMULATE_* config 追加
│   ├── missions/
│   │   ├── stay.py                 # ★ ACCUMULATE hold/fire 追加 (case6 から拡張)
│   │   └── (他は case6 と同一)
│   ├── movements/                  # case6 と同一
│   ├── strategy.py                 # ★ accumulate_fire を SINGLE_SOURCE_MISSION_KINDS に追加
│   └── strategy_helpers.py         # case6 と同一
└── evaluation/
    ├── snapshot_update.py
    └── compare_v6.py               # ★ case7 vs baseline_v6 比較
```

## 完全 ablation スイッチ

`baseline/core/config.py` の `ACCUMULATE_ENABLED = False` で case7 ≡ case6 に戻る。
