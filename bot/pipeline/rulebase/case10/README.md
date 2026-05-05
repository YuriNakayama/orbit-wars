# case10 — baseline_v10 (case7 + accumulate step guard)

case7 (baseline_v7) のフルコピー上に、`_build_accumulate` 冒頭に
`world.step < ACCUMULATE_MIN_LAUNCH_STEP=30` ガードを追加した派生 case。

## 採用戦略

case9 vs case7 の 10戦集計 (2026-05-05) で観察した **t14 罠** を消す改修。
`accumulate_fire` mission が `ACCUMULATE_KNEE_SHIPS=60` 到達時点 (production 高 home なら t13-15 前後) で 60 ships を一斉発射し、敵反撃で大半を喪失するパターンが 70% (7/10) で発生していた。

### 仮説

序盤 (`step<30`) では accumulate を発動しないことで t14 罠 trigger を 0 に下げ、case7 vs `baseline_v4` 勝率を **+20pp 級改善** して ≥55% を達成。中盤以降の accumulate 機能は維持。

詳細: `docs/experiment/rulebase/20260505_case10_accumulate_step_guard/plan.md`
関連 memory: `project_case7_t14_trap.md`

## 成績

- vs baseline_v4 (Stage A sweep 30/50/100×30戦): TBD
- vs baseline_v4 (Stage B 100戦): TBD

## 構造

```
case10/
├── main.py
├── baseline/
│   ├── core/config.py                 # ★ ACCUMULATE_MIN_LAUNCH_STEP=30 追加
│   ├── missions/stay.py               # ★ _build_accumulate 冒頭にガード 1 行追加
│   └── (他は case7 と同一)
└── evaluation/                        # case7 と同一
```

## 完全 ablation スイッチ

`baseline/core/config.py` の `ACCUMULATE_MIN_LAUNCH_STEP = 0` で case10 ≡ case7 に戻る (ガード無効)。
`ACCUMULATE_ENABLED = False` で accumulate 系自体を OFF にできる (case6 等価)。
