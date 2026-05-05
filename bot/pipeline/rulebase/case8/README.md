# case8 — baseline_v8 (case7 + multi-turn beam search optimizer)

case7 (`baseline_v7`, accumulate_burst) のフルコピー上に、greedy mission
selector を **multi-turn beam search optimizer** に置き換えた派生 case。

## 採用戦略

case7 までの mission 列挙 (`missions/`)・movement emit (`movements/`)・
opponent model・lookahead 資産はそのまま再利用しつつ、
`baseline/strategy.py:plan_moves` の「mission を score 降順に 1 個ずつ
greedy commit」していたループを `baseline/planner/` 配下の
beam search に置き換える。

### 仮説

greedy mission selector は (a) ターン内のミッション間で艦数バジェットを
取り合った時の joint 最適性、(b) 自軍の発射が次ターン以降の盤面に与える
影響の 2 軸で局所解に陥っている。これを **multi-turn beam search**
(横断 N=2-3 ターン、beam_width B=4-8) に置換することで、
vs `baseline_v4` (production) のローカル 300 戦勝率を
**+5pp 以上 (≥55%)** 改善できる。

case3 result.md (`docs/experiment/rulebase/20260420_case3_rollout_ablation/`)
が「次の改善には MCTS / beam search が必要」と明記した方向の続編。
詳細: `docs/experiment/rulebase/20260504_case8_multistep_beam/plan.md`

## 成績

- vs baseline_v4 (300戦, seat 入替): TBD (`evaluation/compare_v4.py` で測定)
- vs baseline_v7 (beam の純粋寄与確認用): TBD

## 構造

```
case8/
├── main.py
├── baseline/
│   ├── agent.py                       # case7 と同一
│   ├── core/                          # case7 と同一 + BEAM_* config 追加
│   ├── missions/                      # case7 と同一 (mission 列挙ロジック)
│   ├── movements/                     # case7 と同一
│   ├── lookahead.py                   # case7 と同一
│   ├── opponent_model.py              # case7 と同一
│   ├── strategy.py                    # ★ greedy ループを planner.beam.run へ delegate
│   ├── strategy_helpers.py            # case7 と同一
│   └── planner/                       # ★ 新設サブパッケージ
│       ├── __init__.py
│       ├── beam.py                    # beam search core
│       ├── candidate.py               # mission 部分集合 → MovesPlan の生成・列挙
│       ├── evaluator.py               # plan 評価関数
│       └── simulator.py               # case3 rollout.py から複製した N ターン展開
└── evaluation/
    ├── snapshot_update.py
    └── compare_v4.py                  # ★ case8 vs baseline_v4 比較
```

## 完全 ablation スイッチ

`baseline/core/config.py` の `BEAM_ENABLED = False` で case8 ≡ case7 に戻る
(greedy 経路を踏む)。回帰テスト用。
