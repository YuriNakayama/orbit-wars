# Rulebase/case10 — iter4: Dynamic Threshold for Accumulate

> 作成日: 2026-05-05
> 関連:
> - [`iter1_result.md`](./iter1_result.md) — step guard 単独で 53.0% (n=100)
> - [`iter2_result.md`](./iter2_result.md) — KNEE=40 一律削減は -3pp 逆効果
> - [`iter3_result.md`](./iter3_result.md) — thrash filter は機能 (-36% planet_loss) するも勝率変わらず
> - 30戦敗北分析: Type A 早期消滅 8戦 + Type B 長期消耗 6戦、敗因は accumulate の **need に対し過剰発射** と判明
> スコープ: `_accumulate_target_threshold` を **need-based 動的化**、過剰発射を抑制

## 仮説 (Hypothesis)

case10 iter1-3 の 30戦敗北分析で **真の敗因は accumulate_fire の "need=10 の target にも 60 ships を投げる" 過剰発射** と判明。`_accumulate_target_threshold` が `max(need + SAFETY=4, KNEE_SHIPS=60)` で常に最低 60 ships 要求するため、production 高い home が時間経過で **強制的に 60 ships に到達 → 一斉発射 → 敵反撃で大半喪失** のパターン。

`threshold` を **need ベースで動的化** (`max(need + SAFETY, KNEE_FLOOR=30)` を底、`need × OVERSHOOT_RATIO=1.5` を実質的な上限) すると、small-need target には 30-40 ships で送れるため:
- **発射量が小さい → 敵反撃に対する damage 比が低い**
- 1 source で複数の small target を順次攻撃可能
- 期待効果: vs v4 で **+3-5pp** 改善、case10 を **≥55% に到達**

## 既存コードの現状

- `bot/pipeline/rulebase/case10/baseline/missions/stay.py:_accumulate_target_threshold` (line 299-308):
  ```python
  need = world.ships_needed_to_capture(target_id, arrival_turn, planned_commitments)
  threshold = max(need + cfg.ACCUMULATE_SAFETY_SHIPS, cfg.ACCUMULATE_KNEE_SHIPS)  # ★ floor 60
  ```
- `ACCUMULATE_KNEE_SHIPS=60` は **(a) threshold floor、(b) probe_speed 計算** の 2 箇所で使われる
- 30戦敗北分析: 8/14 LOSS で序盤 (t40-100 帯) に **-100〜-217 ships の連続 ship_loss_burst** を観察、accumulate_fire の発射量が原因

## スコープ (Scope)

### 変更ファイル (2 ファイル)

```
bot/pipeline/rulebase/case10/baseline/
├── core/config.py                 # ★ ACCUMULATE_KNEE_SHIPS_FLOOR + ACCUMULATE_KNEE_OVERSHOOT_RATIO 追加
└── missions/stay.py               # ★ _accumulate_target_threshold を need-based 動的化
```

### config 追加

```python
# core/config.py
# iter4: accumulate threshold の need-based 動的化
# floor 30 = 旧 KNEE_SHIPS=60 の半分。OVERSHOOT_RATIO で過剰送信を抑制。
ACCUMULATE_KNEE_SHIPS_FLOOR: int = 30
ACCUMULATE_KNEE_OVERSHOOT_RATIO: float = 1.5
```

`ACCUMULATE_KNEE_SHIPS=60` は **probe_speed 計算用** に維持 (削除すると速度効率が崩れる)。

### 関数改修

```python
# missions/stay.py:_accumulate_target_threshold (改修前)
need = world.ships_needed_to_capture(target_id, arrival_turn, planned_commitments)
threshold = max(need + cfg.ACCUMULATE_SAFETY_SHIPS, cfg.ACCUMULATE_KNEE_SHIPS)
return threshold

# 改修後
need = world.ships_needed_to_capture(target_id, arrival_turn, planned_commitments)
floor_threshold = max(need + cfg.ACCUMULATE_SAFETY_SHIPS, cfg.ACCUMULATE_KNEE_SHIPS_FLOOR)
overshoot_cap = max(
    need + cfg.ACCUMULATE_SAFETY_SHIPS * 2,
    int(need * cfg.ACCUMULATE_KNEE_OVERSHOOT_RATIO),
)
return min(floor_threshold, overshoot_cap)
```

`overshoot_cap` の `SAFETY_SHIPS * 2` は need が極端に小さい (need=2 で cap=3) 場合のフォールバック。need が 0 の場合 cap=8 で、floor_threshold=30 と比較して 8 が小さい場合は 8 を採用。

## 実装ステップ

1. `core/config.py` に `ACCUMULATE_KNEE_SHIPS_FLOOR=30` と `ACCUMULATE_KNEE_OVERSHOOT_RATIO=1.5` を追加
2. `missions/stay.py:_accumulate_target_threshold` を上記の動的化式に書き換え
3. `tests/pipeline/rulebase/case10/test_accumulate.py` の既存テストが threshold の値を直接 assert している箇所を確認、必要なら修正
4. lint / format / mypy / pytest 緑

## 検証方法

### ローカル

```bash
uv run --directory bot pytest tests/pipeline/rulebase/case10 -m "not slow" -x
```

### 性能評価

```bash
# Stage A: 30戦 + replay 分析
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v10 \
    --mode 1v1 -n 30 --seed 84000 --parallel 4

# 判定:
#   ≥55% → Stage B 100戦 で確認、しきい値達成
#   53% 前後 → iter1 と差なし、iter4 撤退
#   <50% → 動的化が逆効果、iter1 設定に戻す
```

### 副次評価

replay 分析 (Type A LOSS パターン再現の有無):
- t40-100 帯の self ship_loss_burst が **-100〜-217 を超えない** ことを確認
- 同 30戦の Type A LOSS が **8 → 4 以下** に減れば成功

## リスクと早期撤退条件

- **OVERSHOOT_RATIO=1.5 で発射量が需要より少なく capture できない**: target が need=20 なら cap=30、敵反撃で取り返される確率増。Stage A で勝率 <50% なら ratio を 2.0 に上げて再評価
- **floor=30 で速度効率劣化**: KNEE_SHIPS=60 が `fleet_speed` の knee として選ばれていた経緯を踏まえ、30 ships 単発では速度不足の可能性。replay で確認
- **iter2 KNEE=40 と類似で逆効果**: iter2 (一律 KNEE=40) で -3pp になった経験あり。違いは「動的に上限制御」=「small target には 30、large target には need+8 で過剰送らない」。need-based なので case-by-case で異なる
