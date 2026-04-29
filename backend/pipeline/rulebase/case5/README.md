# case5 — baseline_v5 (LB1224 port)

公開 Kaggle ノートブック [orbit-star-wars-lb-max-1224 (Roman Tamrazov)](https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224) の verbatim port。

## 採用戦略

LB1224 ノートブックの 1 ファイル戦略全体を `baseline/agent_full.py` (2455 行) に保持。
Phase A で notebook をそのまま import し、Phase B で pure-helper のみ `baseline/core/{config,types,physics,world_helpers}.py` に分離。

WorldModel + 戦略本体は **意図的に分割しない**。理由:
- 上流 notebook との parity 維持 (バグ追跡・更新時の差分把握)
- ablation 用途中心で production には case4 を使う

## 成績

- vs baseline_v4 自己対戦: 56% 勝率
- vs baseline_v1: 70% 勝率
- publicScore: **600** (case4 = 745 より低い → publicScore は不安定指標、ローカル勝率を優先)
- 詳細: `project_case5_validation.md`

## 構造

```
case5/
├── main.py
├── baseline/
│   ├── agent.py             # agent_full.py を re-export
│   ├── agent_full.py        # 2455 行の notebook port (refactor 対象外)
│   └── core/                # config, types, physics, world_helpers
└── evaluation/
    ├── snapshot_update.py   # src/evaluation 経由
    ├── compare_v1.py        # アドホック比較
    ├── compare_v4.py        # アドホック比較
    └── debug_splits.py      # アドホック
```

## なぜ Strategy 分割しないのか

ユーザー方針「case 完全独立 + 既存 case 機能凍結」と、上流 notebook parity の重要性
(将来の上流更新を取り込む際、内部分割があると diff が大きくなる) から、
case1 / case4 で行った `planner/` 抽出は適用しない。
