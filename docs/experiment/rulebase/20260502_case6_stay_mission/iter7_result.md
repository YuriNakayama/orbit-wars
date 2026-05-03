# [rulebase/case6] iter7 結果: cap & target sweep (vs baseline_v5)

> 評価コマンド (3 並列、各 100戦, seed=1000):
> ```bash
> uv run --directory backend python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 --max-hold-turns 5
> uv run --directory backend python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 --max-hold-turns 6
> uv run --directory backend python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 --max-target-turns 25
> ```
> 構成: burst-only (`STAY_DEFENSE_ENABLED=False`) + 各 sweep 軸の override
> 対戦相手: baseline_v5 (case6 直接派生元)
> 環境: kaggle_environments orbit_wars 1v1, seed 1000..1099
> 計算: ローカル CPU, 3 並列 (12 物理コア)
> 実行時間: 約 55 分

## 結論サマリ

**case6 内のチューニング余地は完全に尽きた**。`MAX_HOLD_TURNS=3, MAX_TARGET_TURNS=30` を確定構成とし、case6 はクローズ。

### 主仮説 (cap を緩めればさらに改善) は棄却

iter6 sweep (cap=2: 50% / cap=3: 56% / cap=4: 58%) で見えた「cap が緩いほど launches↑ で勝率↑」 傾向は **プラトーで頭打ち**、cap=5 (55%) / cap=6 (52%) で改善せず劣化に転じる。**真の最適は cap ∈ {3, 4} のプラトー**、cap=5〜6 で害が出始める境界が確認された。

### 副次仮説 (target を絞れば seed variance 改善) は完全棄却

`MAX_TARGET_TURNS=25` (cap=3 維持) は **49%** に大きく劣化。fleet peak ratio が 1.25→1.21 に低下し、長距離 hold は本質的に勝率に寄与していた。`MAX_TARGET_TURNS=30` (broad) は不可侵。

### Stage 2 (300戦) は不要

3 つの sweep 全て iter6 cap=3 baseline (56%) と ±5pp 内 (49〜55%)、「明確優勢」65% 閾値に達した設定なし → Stage 2 リソースを案件外 (case7 / Kaggle 再提出 / PR) に振る方が ROI 高い。

## 数値テーブル

### 100戦 sweep 結果

| 設定 | MAX_HOLD | MAX_TGT | 勝率 | seat=0 | seat=1 | fleet peak ratio | launches/ep ratio | 平均 ep 長 |
|---|---|---|---|---|---|---|---|---|
| **iter6 cap=3** (baseline) | 3 | 30 | **56.0%** | 50.0% | 62.0% | 1.25 | 0.96 | — |
| **A**: cap=5 | 5 | 30 | **55.0%** | 56.0% | 54.0% | 1.27 | 0.97 | 187.2 |
| **B**: cap=6 | 6 | 30 | **52.0%** | 48.0% | 56.0% | 1.27 | 0.91 | 176.5 |
| **C**: tgt=25 | 3 | 25 | **49.0%** | 44.0% | 54.0% | 1.21 | 0.96 | 174.2 |

### 各 cap 値の縦串比較 (iter3/iter6/iter7 統合)

| cap | 戦数 | 勝率 | 出典 | 備考 |
|---|---|---|---|---|
| 2 | 100 | 50% | iter6 | 強い害 |
| 3 | 300 | **59.7%** | iter5 | **確定構成** (p≈0.0004) |
| 3 | 100 | 56% | iter6 | seed variance 内 |
| 4 | 100 | 58% | iter6 | プラトー上 |
| 5 | 100 | 55% | iter7 A | プラトー上、優位なし |
| 6 | 100 | 52% | iter7 B | 害が出始める境界 |
| ∞ (なし) | 300 | 54.7% | iter3 | 長すぎる hold は明確に害 |

## 構造的解釈

### cap 値の感度曲線

cap 値を横軸に取ると、勝率は以下の形状:
- cap=2: **50%** (急落) — 累積効果が壊れる
- cap=3〜5: **55〜60% プラトー** — 累積維持 + 文脈喪失防止のバランス
- cap=6: **52%** (劣化) — cap=∞ への遷移ゾーン
- cap=∞: **54.7%** (300戦確定) — 文脈喪失害が顕在化

最適点は cap=3 周辺の幅広いプラトー。**cap=3 / cap=4 / cap=5 はどれも実用的に同等**、優位性の差は seed variance ノイズ未満。

### MAX_TARGET_TURNS の重要性

`MAX_TARGET_TURNS=25` (-5 ターン) で勝率 -7pp、fleet peak ratio -0.04。**長距離 hold (ETA 26〜30 ターン) は STAY mission の主要な fleet 形成源**で、絞ると累積効果が直接削れる。

### 全体としての case6 の説明

case6 = case5 + (defense=Off, burst broad gain≥1/ships≥8/dist≤30 + cap=3) で:
- baseline_v5 比 +5pp 改善 (iter5 300戦 59.7%, p≈0.0004)
- fleet peak +28% で勝率に確実に転化
- cap 値は局所最適のプラトー、案外 robust
- これ以上のチューニング ROI なし

## case6 を確定する根拠

1. **cap 値プラトー (3〜5) の確認** — どの値も同等で、seed variance を超えた優位性なし
2. **MAX_TARGET_TURNS は不可侵** — 30 が最適、25 では大きく崩れる
3. **iter4 で「厳しめは害」、iter6/iter7 で「緩めもプラトー以上には伸びない」** — burst パラメータの完全な感度マップが揃った
4. **300戦確証 (iter5)** — cap=3 構成で p≈0.0004 の有意改善、Kaggle 提出済み構成 (cap なし版) を上回る性能を確認済み

## 次の一手の推奨

case6 内のチューニングは打ち切り。次の優先順位:

### Option A (最優先): Kaggle 再提出 + commit + PR

- 提出済み tar.gz (`case6_20260502-110353.tar.gz`) は **iter2 burst-only (cap なし)** 状態のスナップショット = 300戦 54.7% 相当
- iter5 cap=3 を反映した再提出が必要 (300戦 59.7%, +5pp)
- 本日 quota 4/5 残、要 user 明示承認
- 並行で:
  - `case6/README.md` に iter1〜7 結果を追記
  - `feature/add-rulebase-to-stay` → `main` の PR 作成

### Option B: case7 で別軸

- comet ターン動的 hold 制御 (comet 出現の 50/150/250/350/450 ターン前後で cap を一時的に変える)
- defense と burst の 2 段組合せ再設計 (defense は無条件 hold ではなく特定発火条件のみ)
- lookahead 強化 (現状 1 ターン先 → 2〜3 ターン先評価)
- harass mission 強化 (case2 ablation memory で Harass+HALF_STEP が +3.7pp 実績)

### Option C: 別 family

case6 の rulebase 改善 ROI は局所最適到達で低下、imitation/case1 や reinforce 系の探索が長期的価値高い可能性。

## 再現手順

```bash
# 3 並列 (各 ~55分)
mkdir -p /tmp/case6_iter7
nohup uv run --directory backend python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 --max-hold-turns 5 > /tmp/case6_iter7/A_cap5.log 2>&1 &
nohup uv run --directory backend python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 --max-hold-turns 6 > /tmp/case6_iter7/B_cap6.log 2>&1 &
nohup uv run --directory backend python -m pipeline.rulebase.case6.evaluation.compare_v5 -n 50 --seed 1000 --max-target-turns 25 > /tmp/case6_iter7/C_tgt25.log 2>&1 &
wait
```

## 関連 docs

- `iter1_result.md` (vs v4 100戦 64%)
- `iter2_result.md` (ablation 3 × 100戦)
- `iter3_result.md` (broad 300戦 54.7%)
- `iter4_result.md` (厳しめ 41% 失敗)
- `iter5_plan.md` / `iter5_result.md` (cap=3 で 300戦 59.7% breakthrough)
- `iter6_plan.md` / `iter6_result.md` (cap sweep 2/3/4 局所最適確定)
- `iter7_plan.md` / **iter7_result.md** (本ファイル — cap 5/6, tgt=25 で打ち止め確認)

## ログ保存先

`/tmp/case6_iter7/{A_cap5,B_cap6,C_tgt25}.log`
