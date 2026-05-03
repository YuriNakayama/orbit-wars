# [rulebase/case6] iter5: burst hold 上限ターン数の導入

## 仮説

iter4 (厳しめ burst) の崩壊から得た最大の知見:
**「broad burst の hold は累積効果で勝率を作るが、同じ source が hold し続けると文脈を失った場面で固まる」**

iter4 で「hold 発火頻度を絞る」方向は失敗したので、iter5 では **「個々の hold の発火条件は broad のまま、ただし同じ source が連続で hold できるターン数に上限を入れる」** というアプローチを試す。これにより:
- iter3 broad の累積効果 (fleet peak 1.28x) は維持
- 「永続 hold」による文脈喪失を防ぐ
- artifact (大艦隊化) を outcome (勝率) に転化させる試み

## 変更点

### 1. 新パラメータ (config.py)

```python
# Maximum consecutive turns a single source may be held by burst. After this,
# the source is forced to launch (or whatever strategy.py decides) for at least
# one turn before it can be held again. Prevents "stuck holds" where the same
# source accumulates indefinitely while context shifts around it.
STAY_BURST_MAX_HOLD_TURNS: int = 3
```

### 2. turn 跨ぎ state (agent.py)

既存 `_OM_STATE` と同じモジュールレベル state パターンで `_STAY_STATE` を追加:

```python
@dataclass
class StayState:
    consecutive_holds: dict[int, int]  # src_id -> turns currently held in a row

_STAY_STATE: StayState = StayState(consecutive_holds={})
```

`agent(obs)` で `_STAY_STATE` を `world` か `plan_moves` に注入。

### 3. build_stay_holds の拡張 (stay.py)

```python
def build_stay_holds(
    world: WorldModel,
    consecutive_holds: dict[int, int] | None = None,
) -> tuple[dict[int, int], list[StayDecision]]:
    ...
    if consecutive_holds is not None:
        # Drop burst holds for sources that have already held MAX_HOLD_TURNS
        # consecutive turns — they need to launch this turn.
        burst_holds = {
            src_id: ships
            for src_id, ships in burst_holds.items()
            if consecutive_holds.get(src_id, 0) < cfg.STAY_BURST_MAX_HOLD_TURNS
        }
    ...
```

### 4. consecutive_holds の更新 (agent.py / plan_moves)

`plan_moves` (or agent.py) で:
```python
new_consecutive = {}
for src_id in stay_holds.keys():
    new_consecutive[src_id] = consecutive_holds.get(src_id, 0) + 1
# Sources not held this turn reset to 0 (omitted from dict).
_STAY_STATE.consecutive_holds = new_consecutive
```

defense holds は対象外 (defense は脅威に対する応答で、文脈喪失は起きにくい)。burst holds のみ追跡。

## 期待効果

- broad burst の累積効果は概ね維持 (発火条件は同じ)
- 同じ source が 4 ターン以上 hold していたケースが消える (推定 hold 発火の 10〜20%)
- launches/ep ratio が **0.97 → 1.00 〜 1.05** に微増
- fleet peak ratio は **1.28 → 1.20 〜 1.25** にやや低下 (許容)
- 勝率が **54.7% → 56〜60%** に上昇する想定

下振れシナリオ:
- 4 ターン以上 hold は実は fleet 形成の本体だった → 勝率低下 (iter4 の縮小版)
- 効果なし (54.7% ±5.6pp の幅で動かない)

## 評価方針

### Stage 1: 100戦 vs v5 (~50分)

```bash
uv run --directory bot python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000
```

判定:
- **57%+** → 改善、Stage 2 (300戦) に進む
- **52〜56%** → seed variance 内、Stage 2 で確証
- **52%未満** → 上限 hold は逆効果、別 case か別軸へ

### Stage 2: 300戦 (Stage 1 で改善が見えたら)

iter3 と同じ 3 並列 seed 1000/2000/3000。

## 非ゴール

- defense の再有効化はしない (iter2 で有害確定)
- burst パラメータ (gain/ships/dist) は iter3 broad のまま (`1, 8, 30`)
- case7 は立てない
- defense にも上限 turn 数を入れない
- 提出済み Kaggle archive は触らない

## 実装範囲

| ファイル | 変更 |
|---|---|
| `case6/baseline/core/config.py` | `STAY_BURST_MAX_HOLD_TURNS = 3` 追加 |
| `case6/baseline/missions/stay.py` | `build_stay_holds` に optional 引数追加、burst フィルタ追加 |
| `case6/baseline/agent.py` | `_STAY_STATE` モジュール global、agent 関数で更新 |
| `case6/baseline/strategy.py` | `plan_moves` で `consecutive_holds` を引き渡し |
| `case6/tests/test_stay_decision.py` | 上限ロジックの単体テスト追加 |
