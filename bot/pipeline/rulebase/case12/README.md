# case12 — baseline_v12 (case4 + Naïve MCTS / CMAB sampling)

case4 (LB745 production) のフルコピー上に、**Combinatorial Multi-armed
Bandit (CMAB) ベースの Naïve MCTS** を `baseline/planner/` 配下に追加した
派生 case。

## 採用戦略

case4 base 全資産を維持しつつ、`strategy.plan_moves` で以下に分岐:
- `NAIVE_MCTS_ENABLED=True`: 各 source に script を **stochastic UCB1
  sampling** で割り当て、組合せ全体を rollout 評価して最頻 assignment を出力
- `NAIVE_MCTS_ENABLED=False`: case4 の greedy mission ordering 経路 (回帰用)

case11 PGS の **「deterministic hill climb による local optimum 固定 (0%)」**
構造問題を sampling-based で bypass する試行。

詳細: `docs/experiment/rulebase/20260505_case12_naive_mcts/plan.md`
関連 memory: `project_heuristic_search_saturation.md`

### 仮説

PGS の hill climb は各 source 独立で local optimum に固定 (case11 で
`script_idle` 主体に陥った)。Naïve sampling は UCB1 で stochastic に組合せ
を探索、PGS が見ない multi-script combination を試行できる。

期待: vs `baseline_v4` で ≥40% (PGS の 0% から大幅改善)、可能なら ≥50%。

## 成績

- iter1 (NaïveMCTS v0) vs baseline_v4: TBD
- 関連 memory: heuristic 系 10 連敗、本実験で 11 番目

## 構造

```
case12/
├── main.py
├── baseline/
│   ├── ...                            # case4 と同一資産
│   ├── core/config.py                 # ★ NAIVE_MCTS_* 追加
│   └── planner/                       # ★ 新設 (case11 から複製 + naive_mcts.py)
│       ├── scripts.py                 # case11 v3 から複製
│       ├── evaluator.py               # case11 から複製
│       └── naive_mcts.py              # ★ 新規 NaïveMCTS 実装
└── evaluation/                        # case4 と同一
```

## 完全 ablation スイッチ

`baseline/core/config.py` の `NAIVE_MCTS_ENABLED = False` で case12 ≡ case4 に戻る。
