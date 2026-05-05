# case9 — baseline_v9 (case4 production + thrash filter)

case4 (LB745 production) のフルコピー上に、case8 iter3 v1 で実証した
**planet thrash filter** (recently_lost only、`THRASH_REPEAT_COMMIT_LIMIT=999`
で commits 経路は無効化) を被せた派生 case。

## 採用戦略

- case4 base 全資産 (capture/snipe/swarm/harass/reinforcement/crash_exploit/
  fleet_consolidation、SAFE_INTERCEPT_HALF_STEP) を維持
- case8 iter3 v1 から `THRASH_*` config + `_update_thrash_state` +
  `apply_score_modifiers` の thrash decay を移植
- BEAM / STAY / accumulate は case4 base に元々無く、本実験でも追加しない

### 仮説

case8 iter1/2/3 が vs `baseline_v4` で ~30% に collapse した主因は
**case7 base 自体の弱さ** (smoke で v4 vs v7 = 60-40)。case4 (LB745)
を base に thrash filter を被せれば、base 起因の handicap (~10pp) が消えて
vs v4 で **≥55%** を達成できる。

詳細: `docs/experiment/rulebase/20260505_case9_thrash_filter_on_case4/plan.md`

## 成績

- vs baseline_v4 (200戦, seat 入替): TBD
- vs baseline_v8 (case7 base + filter, smoke 30戦): TBD (副次評価)

## 構造

```
case9/
├── main.py
├── baseline/
│   ├── agent.py                       # ★ StayState (thrash 用) + _update_thrash_state 追加
│   ├── core/
│   │   ├── config.py                  # ★ THRASH_* 4 個追加
│   │   └── world_model.py             # ★ recently_lost / mission_commits 引数追加
│   ├── missions/                      # case4 と同一 (fleet_consolidation 含む)
│   ├── movements/                     # case4 と同一
│   ├── strategy.py                    # case4 と同一 (BEAM 不要)
│   └── strategy_helpers.py            # ★ apply_score_modifiers に thrash decay 追加
└── evaluation/                        # case4 と同一
```

## 完全 ablation スイッチ

`baseline/core/config.py` の `THRASH_FILTER_ENABLED = False` で case9 ≡ case4 に戻る
(filter 経路を踏まない)。回帰テスト用。
