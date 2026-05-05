# case9 — anti_ping_pong (case4 + ping-pong suppression)

case4 (baseline_v4 = production) のフルコピー上に、近隣惑星間で起こる
**planet ping-pong** (各ターンで小規模 ship を相互送出する振動現象) を
診断・抑制する仕組みを追加した派生 case。

## 採用戦略

case4 (baseline_v4, fleet_consolidation 入り) を起点に、以下を追加:

1. **dispatch 履歴 persistence** — `agent.py` の module-level に
   `_DISPATCH_HISTORY` を持ち、各ターンの `(src_id, est_dst_id) → step`
   を記録。新規ゲーム検出時にクリア。
2. **Pair cooldown** — `missions/reinforcement.py` で同 `(src, dst)` pair が
   `PING_PONG_PAIR_COOLDOWN_TURNS` 以内に発火していたら skip。
3. **Harass target cooldown** — `missions/harass.py` で同 target に
   `HARASS_TARGET_COOLDOWN_TURNS` 以内に harass 済みなら skip。
4. **`REINFORCE_MIN_DEFICIT`** — `_compute_defense_buffers` の
   `threatened_candidates` 構築で `deficit_hint < REINFORCE_MIN_DEFICIT` を除外。

## 仮説

ping-pong は (a) reinforce 発火閾値が低すぎる (b) harass がクールダウン無し
(c) `plan_moves` が履歴非依存、の 3 点が複合して起きている。これらを抑制
した分の余剰 ship を ACCUMULATE / multi-source swarm / rear-guard に
流用することで vs baseline_v4 勝率改善 (+5pp 以上) を狙う。

詳細: `docs/experiment/rulebase/20260504_case9_anti_ping_pong/plan.md`

## 成績

- vs baseline_v4 (300戦): TBD (`evaluation/compare_v4.py` で測定)
- ping-pong 件数削減率: TBD (`evaluation/diagnose_ping_pong.py` で測定)

## 構造

```
case9/
├── main.py
├── baseline/
│   ├── agent.py                       # ★ _DISPATCH_HISTORY を保持
│   ├── core/
│   │   ├── config.py                  # ★ ANTI_PING_PONG_* 定数を追加
│   │   └── world_model.py             # ★ recent_dispatches を WorldModel に
│   ├── missions/
│   │   ├── reinforcement.py           # ★ pair cooldown
│   │   └── harass.py                  # ★ target cooldown
│   ├── movements/                     # case4 と同一
│   ├── strategy.py                    # case4 と同一
│   └── strategy_helpers.py            # case4 と同一
└── evaluation/
    ├── snapshot_update.py
    ├── ablation.py
    ├── diagnose_ping_pong.py          # ★ ping-pong 件数を計測
    └── compare_v4.py                  # ★ case9 vs baseline_v4 比較
```

## 完全 ablation スイッチ

`baseline/core/config.py` の `ANTI_PING_PONG_ENABLED = False` で
case9 ≡ case4 に戻る (cooldown / deficit 閾値はすべて bypass)。
