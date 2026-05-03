# case2 — baseline_v2

case1 の後継で、opponent model + lookahead + harass / swarm mission を追加。

## 採用戦略

- `baseline/missions/` 拡張: harass.py, swarm.py 新設
- `baseline/movements/` 新設: evacuation, followup, rear_guard
- `baseline/opponent_model.py`, `lookahead.py` 追加

## 成績

- ablation 結果は memory `project_case2_ablation.md` 参照。100戦は seed variance が大きく、Harass+HALF_STEP は 300戦で +3.7pp (非有意)。
- COMET_NPV と FINISHING_TIE_GUARD は害があるため OFF (デフォルト)。

## 構造

```
case2/
├── main.py
├── baseline/
│   ├── agent.py, strategy.py, strategy_helpers.py
│   ├── core/         # types, geometry, physics, world_model
│   ├── missions/     # snipe, crash_exploit, reinforcement, harass, swarm, capture
│   ├── movements/    # evacuation, followup, rear_guard
│   ├── opponent_model.py
│   └── lookahead.py
├── configs/baseline.yaml
└── evaluation/{snapshot_update.py, ablation.py}
```
