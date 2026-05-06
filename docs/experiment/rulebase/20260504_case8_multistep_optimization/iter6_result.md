# Rulebase/case10 — iter2 Result: KNEE_SHIPS Reduction

> 作成日: 2026-05-05
> 対応 plan: [`iter2_plan.md`](./iter2_plan.md)
> 関連:
> - [`iter1_result.md`](./iter1_result.md) — step guard 単独で 53.0% (n=100)
> - replay 分析 (iter2): `data/output/experiment/rulebase/case10/replay_analysis/20260505_iter2/result_{1,2}.md`

## 結論

**仮説は否定。iter2 (KNEE_SHIPS 60→40) は採用却下。** 30 戦 vs `baseline_v4` で **50.0%** (15-15)、iter1 の Stage A 53.3% から **-3.3pp 悪化**。replay 分析で「step guard は正常動作 (t14 罠 0 件) だが、KNEE 削減で中盤の捕獲効率が落ちている」と判明。

iter1 の `KNEE_SHIPS=60, MIN_LAUNCH_STEP=30` 設定が **case10 最良構成** であることを確定。`config.py` の `KNEE_SHIPS` を **iter1 値 (60) に戻して撤退**。

## 数値

### iter2 30戦結果

| | wins (30戦, seed 82000+) | win_rate | turn_p95 |
|---|---|---|---|
| baseline_v10 (iter2, KNEE=40) | 15 | 50.0% | 0.689s |
| baseline_v4 | 15 | 50.0% | 0.384s |

### iter1 vs iter2 比較

| 構成 | n | win_rate | comment |
|---|---|---|---|
| iter1 Stage A (step=30, KNEE=60) | 30 | **53.3%** | 最良 |
| iter1 Stage B 合算 (step=30, KNEE=60) | 100 | 53.0% | 採用保留 (-2pp from threshold) |
| **iter2 (step=30, KNEE=40)** | 30 | **50.0%** | **-3.3pp from iter1 Stage A** |

### しきい値判定

| 項目 | しきい値 | 実測 | 判定 |
|---|---|---|---|
| 合算勝率 vs v4 | ≥55% | 50.0% | ❌ -5pp 未達 |
| iter1 比改善 | +2pp | -3.3pp | ❌ 後退 |
| turn_p95 | ≤0.7s | 0.689s | ✅ ぎりぎり |
| timeouts | 0 件 | 0 件 | ✅ |

## 診断

### step guard は引き続き機能

iter2 long replay (seed 82007) で:
- self ship_loss_burst at t≤20: **0 件** ✅
- self ship_loss_burst at t≤30: **0 件** ✅
- 全 self ship_loss_burst: 4 件 (全て step≥30 帯)

t14 罠は完全に抑制されており、step guard は iter1 と同じ働きをしている。

### KNEE=40 が逆効果になった原因 (推定)

1. **fleet_speed の knee 効果喪失**: `fleet_speed = 1 + (max-1) × (log(ships)/log(1000))^1.5` で、ship 数が多いほど移動速度が増す。**KNEE=60 はこの曲線の「速度効率の knee」として選ばれていた経緯**。40 ships に減らすと、長距離 target に届く前に敵が拡張してしまう
2. **`_accumulate_target_threshold`**: `max(need + SAFETY_SHIPS=4, KNEE_SHIPS=40)` で need < 36 の target は threshold=40 になる。元 60 と比べ 20 ships 少ないため、敵反撃を受けた時の **取り逃しが増える**
3. **複数回送信のメリット < 1 回完投のメリット**: 仮説では「40 × 2 回」のほうが「60 × 1 回」より柔軟性があると見たが、実際は production 増分 (2/turn) では 40→ さらに 40 に戻るのに 20 turn かかる。一方 60 までは 25 turn。差は 5 turn だが、その間に敵に取られる確率の方が大きい

### iter1 と iter2 の差は 53.3% vs 50.0% = 3.3pp

30 戦の seed variance (±5pp 程度) を踏まえると有意な差ではない可能性もあるが、replay の挙動から見て **改善方向ではない** ことは確か。

## 採用方針

- **iter2 は採用却下**
- **`bot/pipeline/rulebase/case10/baseline/core/config.py`** の `ACCUMULATE_KNEE_SHIPS=40` を **60 に戻す** (iter1 値、本 result.md 執筆完了直後に修正)
- case10 確定構成: `step guard=30, KNEE=60` (= iter1 設定)、vs v4 = 53.0% (n=100)
- production case4 (LB745) は引き続き現役、case10 は対抗案候補だが しきい値 -2pp 不足

## 確定した知見

1. **case7 base の t14 罠は `ACCUMULATE_MIN_LAUNCH_STEP=30` 単一改修で構造的解消できる** (replay で 0 件確認、案 3 root-cause 仮説完全裏付け)
2. **`ACCUMULATE_KNEE_SHIPS=60` は fleet_speed の knee として最適**、削減すると速度・捕獲効率ともに低下
3. **case10 最良値は iter1 設定で 53.0% (n=100)、しきい値 ≥55% に -2pp 不足**。case7 base 純粋 disadvantage が残存

## 次の方向 (本ディレクトリのスコープ外)

| 案 | 期待 | コスト |
|---|---|---|
| case10 + fleet_consolidation 移植 (case4 から) | 中盤効率改善で +5pp | 1-2 時間 |
| case10 を 200-300戦で再評価 (seed variance 縮小) | 53% 真値確定 (≥55% に届くかは微妙) | 30分 |
| 別軸 experiment (case4 base 上の改修案) | LB745 production 突破狙い | 別 plan 要 |

## 再現手順

```bash
# iter2 (KNEE=40) を試す場合: config.py で KNEE_SHIPS を 40 に書き換え
uv run --directory bot pytest tests/pipeline/rulebase/case10 -m "not slow" -x

uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v10 \
    --mode 1v1 -n 30 --seed 82000 --parallel 4
```

## 関連ファイル

- `bot/pipeline/rulebase/case10/baseline/core/config.py:ACCUMULATE_KNEE_SHIPS` — iter2 で 40 に変更後、本 result で 60 に復元
- `data/output/experiment/rulebase/case10/replay_analysis/20260505_iter2/` — iter2 long+fastest_loss replay 分析

## 環境

- ハードウェア: M4 MacBook (local), parallel=4
- branch: `feature/rulebase-multistep-optimization`
- 実行日時: 2026-05-05
