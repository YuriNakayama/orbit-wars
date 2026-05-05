# case11 — baseline_v11 (case4 + Portfolio Search family)

case4 (LB745 production) のフルコピー上に、**source 単位で script を割り当てる
portfolio search 系手法** を 3 段階で iter 単位に検証する派生 case。

## 採用戦略

case4 base 全資産 + **Portfolio Greedy Search (PGS)** / Nested-Greedy Search
(NGS) / Naïve MCTS (CMAB) を `baseline/planner/` 配下に追加。

- iter1 = PGS (本 commit で実装)
- iter2 = NGS (iter1 の結果次第で実施)
- iter3 = NaïveMCTS (iter1/2 で ≥55% 未達なら最終手段)

詳細: `docs/experiment/rulebase/20260505_case11_portfolio_search/plan.md`

### 仮説

case8 (beam) と case10 (heuristic 改修) で確認した **「mission ordering 補正/
並び替え」方針の飽和** を、**「source 単位の script 割当」探索** という直交軸
で打破する。RTS AI 文献 (Churchill & Buro 2013、Moraes 2018、Ontañón 2017) で
実績のある portfolio search 系手法を本実装で評価。

## 成績

- iter1 (PGS) vs baseline_v4: TBD
- iter2 (NGS) vs baseline_v4: 任意 (iter1 ≥55% なら skip)
- iter3 (NaïveMCTS) vs baseline_v4: 任意 (iter1/2 ともに <55% なら最終手段)

## 構造

```
case11/
├── main.py
├── baseline/
│   ├── ...                            # case4 と同一資産
│   ├── core/config.py                 # ★ PORTFOLIO_*, NGS_*, NAIVE_MCTS_* を追加
│   └── planner/                       # ★ 新設
│       ├── scripts.py                 # 7 個の script (idle/capture_safe/...)
│       ├── evaluator.py               # playout-based value function
│       ├── portfolio_greedy.py        # iter1: PGS
│       ├── nested_greedy.py           # iter2: NGS (iter1 後)
│       └── naive_mcts.py              # iter3: NaïveMCTS (iter2 後)
└── evaluation/                        # case4 と同一

case4 base から継承:
- baseline/missions/{capture, snipe, swarm, harass, reinforcement, crash_exploit}
- baseline/movements/{followup, evacuation, rear_guard}
- baseline/core/physics.py の SAFE_INTERCEPT_HALF_STEP
```

## 完全 ablation スイッチ

`baseline/core/config.py` の `PORTFOLIO_ENABLED = False` で case11 ≡ case4 に戻る。
回帰テストで保証 (`tests/pipeline/rulebase/case11/test_pgs_off_equals_case4.py`)。
