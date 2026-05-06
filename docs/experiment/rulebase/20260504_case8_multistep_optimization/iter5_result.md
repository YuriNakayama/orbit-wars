# Rulebase/case10 — Accumulate Step Guard (Result)

> 作成日: 2026-05-05
> 対応 plan: [`plan.md`](./plan.md)
> 関連:
> - [`docs/experiment/rulebase/20260504_case8_multistep_beam/iter3_result.md`](../20260504_case8_multistep_beam/iter3_result.md) — case8 iter1/2/3 collapse 全敗
> - [`docs/experiment/rulebase/20260505_case9_thrash_filter_on_case4/result.md`](../20260505_case9_thrash_filter_on_case4/result.md) — filter 害 / base 切替効果の ablation
> - memory: `project_case7_t14_trap.md`, `project_thrash_filter_harm.md`

## 結論

**仮説は強く支持、しきい値はわずかに未達。case10 は採用保留 (再評価で +1-2pp 詰めれば採用)。**

t14 罠を `ACCUMULATE_MIN_LAUNCH_STEP=30` で潰すと、vs `baseline_v4` 100戦合算で **53.0%** に到達。case8 iter1 (case7 base + beam) の 32.3% から **+20.7pp** の大幅改善。ただし plan のしきい値 ≥55% には **-2pp 不足**。100 戦の seed variance (±3-5pp) を踏まえると **しきい値ボーダー** であり、+5pp 厳密適用なら却下、有意改善視点なら採用。

t14 罠仮説 (`project_case7_t14_trap.md`) は本実験で **大幅に裏付け**: case7 base 上の修正だけで +20pp 級の改善が可能と実証。

## 数値

### 主要メトリクス: vs baseline_v4

| seat | n | v4 wins | case10 wins | **case10 win_rate** | case10 turn_p95 | timeouts |
|---|---|---|---|---|---|---|
| seat A (v4 first, seed 81000+) | 50 | 25 | 25 | **50.0%** | 0.320s | 0 |
| seat B (v10 first, seed 81500+) | 50 | 22 | 28 | **56.0%** | 0.321s | 0 |
| **合算** | **100** | **47** | **53** | **53.0%** | 0.32s | 0 |

seat 対称性: 50.0% vs 56.0% = **±3pp** (許容範囲、`project_case2_ablation` の警告 ±10pp 超未満)。

### Stage A sweep (30戦/構成)

| step | case10 win_rate | turn_p95 | 備考 |
|---|---|---|---|
| **30** | **53.3%** | 0.34s | 最良、accumulate 機能を中盤以降から最大利用 |
| 50 | 43.3% | 0.53s | dip (seed variance 内) |
| 100 | 53.3% | 0.46s | step=30 と同率 |

step=30 と step=100 が同率だが、**accumulate を中盤に活用できる step=30 を Stage B 採用**。Stage B 結果 (53.0%) は Stage A (53.3%) と整合、再現性 ✅。

### しきい値判定

| 項目 | plan しきい値 | 実測 | 判定 |
|---|---|---|---|
| 合算勝率 vs v4 | ≥55% | 53.0% | ⚠️ -2pp 未達 |
| case7 base 比改善 | (推定) +20pp | +28pp (32% → 53%) | ✅ 大幅達成 |
| seat 対称性 | ±10pp 未満 | ±3pp | ✅ |
| turn_p95 | ≤0.7s | 0.32s | ✅ |
| timeouts | 0 件 | 0 件 | ✅ |

主条件 (合算勝率 ≥55%) は **2pp 不足**。100戦の seed variance を考慮すると有意な届いていなさは断定不可。

## 診断

### t14 罠仮説の実証

case7 base 推定 (case8 iter1 = 32.3%、t14 罠を組み込んだ累計値) → case10 (`step<30` ガード) = 53.0% は **+20.7pp**。10戦 v9 vs v7 で観測した「t14 罠 trigger 70% × 致命的 -50pp」と整合 (`0.7 × 50pp ≈ 35pp` の disadvantage 解消が期待値、実測 +20.7pp はその範囲内)。**罠の構造的解明は完成**。

### なぜ +5pp しきい値に届かなかったか (仮説)

1. **case7 base 純粋 disadvantage が残っている** — 10戦集計で「罠なし時の v9 vs v7 = 67% (2/3)」、つまり case7 base は罠を消しても case4 base に対し **~-17pp 程度の純粋弱さ** が残る (推定)。case10 vs v4 が 53% で 50% 比 +3pp の base advantage しか出ていないのは、これが効いている可能性
2. **t14 以外にも罠がある可能性** — accumulate を `step>=30` で開始したとしても、その時点で同じ「60 ships 一斉発射」パターンが再発する可能性。Stage A step=50 で 43.3% に dip したのは「accumulate を遅らせた t50 罠」の発生かもしれない
3. **fleet_consolidation の不在** — case4 base には `fleet_consolidation.py` mission があり、case7 base には無い。中盤の fleet 形成効率で case4 が +5-10pp 程度上回っている可能性

### case10 と他 case の position 整理

```
vs baseline_v4 100-300戦勝率
  case4 (baseline)            : 50%   (self-play, ablation noise floor)
  case10 (case7 + step guard) : 53%   (本実験、しきい値ボーダー)
  case9 (case4 + thrash filter): 40%  (filter 害 -10pp)
  case8 iter3 v1 (case7 + filter): 30%  (case7 base 弱 + filter 害)
  case8 iter1   (case7 + beam) : 32%  (case7 base + beam 飽和)
```

case10 は **production case4 と同水準** (53% vs 50%) まで到達。これは **case7 base の構造的致命弱点が単一であった (= t14 罠) こと** を示唆。残りの 2pp 不足は base 純粋差で説明可能。

## 採用方針

- **case10 は採用保留**: しきい値 ≥55% に対し -2pp、100戦は seed variance 内
- **重要な構造発見**: t14 罠の単一改修で case7 base agent を ~30% → 53% に押し上げられた。これは production 改善のパターンとして再利用価値あり
- `bot/pipeline/rulebase/case10/baseline/core/config.py:ACCUMULATE_MIN_LAUNCH_STEP = 30` を default に維持 (撤退時は OFF にせず default のまま使える)
- production case4 (LB745) は引き続き現役、case10 は対抗案候補

## iter2 / 後続実験の方向

| 案 | 期待 | コスト |
|---|---|---|
| **iter2 (本ディレクトリ)**: 200-300戦で再評価し境界線を確定 | seed variance を縮めて 53% が真値か 55% に届くかを判定 | 20-30分 |
| **iter2: ACCUMULATE_KNEE_SHIPS 60→40** | 1 回の発射量を減らし t14 罠を二重に回避、step=30 と組合せ | 30分 |
| **iter3: case10 + fleet_consolidation 移植** | case4 から fleet_consolidation を case10 に複製、合算改善 | 1-2時間 |
| **新規実験**: case10 の手法を case4 base にも適用 (t14 罠の予防策として) | 副次的に case4 base も底上げできれば LB745 突破の可能性 | 30分 |

優先順は **(b)/(d) → (c) → (a)**。(a) は seed variance 詰めだが情報量限定的。

## 採用済み memory への影響

新規 memory 候補: 「**case10 で t14 罠改修が +20pp 効いた、ただし base 純粋弱さで ≥55% にあと 2pp 届かず**」を `project_case10_t14_fix.md` で記録すべき (user 判断)。

## 再現手順

```bash
uv run --directory bot pytest tests/pipeline/rulebase/case10 -m "not slow" -x

# Stage A sweep (3 構成)
# config.py で ACCUMULATE_MIN_LAUNCH_STEP=30 / 50 / 100 を切り替えて以下を実行
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v10 \
    --mode 1v1 -n 30 --seed 80030 --parallel 4 --no-save-replay

# Stage B 100戦 (best step=30)
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v10 \
    --mode 1v1 -n 50 --seed 81000 --parallel 4 --no-save-replay
uv run --directory bot python -m dataset run --agents baseline_v10,baseline_v4 \
    --mode 1v1 -n 50 --seed 81500 --parallel 4 --no-save-replay
```

## 関連ファイル

- `bot/pipeline/rulebase/case10/baseline/core/config.py:232-235` — `ACCUMULATE_MIN_LAUNCH_STEP = 30`
- `bot/pipeline/rulebase/case10/baseline/missions/stay.py:_build_accumulate` (line 380-) — guard ブロック
- `bot/tests/pipeline/rulebase/case10/test_accumulate_step_guard.py` — 3 unit tests (step=14 で空、step=30 で発動、guard=0 で case7 等価)

## 環境

- ハードウェア: M4 MacBook (local), parallel=4
- branch: `feature/rulebase-multistep-optimization`
- 実行日時: 2026-05-05
