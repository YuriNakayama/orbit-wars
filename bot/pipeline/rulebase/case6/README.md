# case6 — baseline_v6 (case4 + STAY judge)

case4 (production champion baseline_v4) のフルコピー上に、
ship を発射せずに保留する **STAY 判定** を追加した派生 case。

## 採用戦略

case4 (= sigmaborov LB897 reinforce port + fleet_consolidation) の構成に、
`baseline/missions/stay.py` を追加。STAY は mission として個別に
コミットされるのではなく、`plan_moves` 冒頭で per-source の
**hold 量** を計算し、`source_attack_left` をラップすることで
capture / swarm / harass / followup / rear_guard 全てに反映される。

### 二目的

1. **Defense hold** — 既に飛来中の敵 fleet を踏まえ、近 horizon
   (`STAY_DEFENSE_HORIZON = 12`) で自惑星の駐留が割れる risk を
   `value` で重み付け、閾値超過なら src で ship を留め置く。
2. **Burst hold** — 1 ターン待って production を加算してから
   発射すれば `fleet_speed = 1 + (max-1) × (log(ships)/log(1000))^1.5`
   が上昇し ETA が短縮するケースで、その差が
   `STAY_BURST_MIN_GAIN >= 1` ターン以上なら src を 1 ターン留置。

詳細: `docs/experiment/rulebase/20260502_case6_stay_mission/plan.md`

## 成績

- vs baseline_v4 (100 戦): TBD (`evaluation/compare_v4.py` で測定)

## 構造

```
case6/
├── main.py
├── baseline/
│   ├── agent.py
│   ├── core/                       # case4 と同一 + STAY_* config 追加
│   ├── missions/
│   │   ├── stay.py                 # ★ 追加: build_stay_holds
│   │   └── (他は case4 と同一)
│   ├── movements/                  # case4 と同一
│   ├── strategy.py                 # ★ stay_holds で source_attack_left をラップ
│   └── strategy_helpers.py         # case4 と同一
└── evaluation/
    ├── snapshot_update.py
    └── compare_v4.py               # ★ case6 vs baseline_v4 比較
```

## 完全 ablation スイッチ

`baseline/core/config.py` の `STAY_ENABLED = False` で case6 ≡ case4 に戻る。
`STAY_DEFENSE_ENABLED` / `STAY_BURST_ENABLED` で個別に切り替え可能。
