# imitation/case5 — Ship-Prediction Featurizer (cycle 3 進捗)

> 作成日: 2026-05-03
> plan.md: `./plan.md`
> Cycle 3 スコープ: plan.md 作成 + 既存 rulebase ship-prediction の効力確認 (実装は次サイクル送り)

## Cycle 3 結論

**ship-prediction (`simulate_planet_timeline`) は rulebase 経路で実証済み**。imitation 取り込みに進む価値あり。

### 確認した既存ベンチマーク (`docs/experiment/rulebase/20260502_case6_stay_mission/`)

case6 baseline (`world.base_timeline` を STAY 判定で消費する rulebase agent) を 100-300 ep 自己対戦した既存結果:

| 対戦相手 | 設定 | 戦数 | 勝率 | 出典 |
|----------|------|------|------|------|
| baseline_v4 (production champion) | Full (defense+burst hold) | 100 | **64.0%** | iter1 |
| baseline_v5 (LB1224 port) | Full | 100 | 53.0% | iter2 |
| baseline_v5 | burst-only | 100 | **59.0%** | iter2 |
| baseline_v5 | burst cap=3 | **300** | **59.7%** (p≈0.0004) | iter5 |
| baseline_v5 | burst cap=5 | 100 | 55.0% | iter7 |

**鍵**:
- `STAY_DEFENSE_ENABLED` (timeline-driven defense hold) は v5 相手では効果薄、burst hold が主役
- それでも timeline 由来の min_owned / keep_needed は burst hold の前提条件として必要
- production champion (v4) には ship-prediction 利用版 (case6) が **64%** という非常に強い差をつけている

### Imitation 取り込みの妥当性

**ship-prediction は rulebase で実装済みかつ強力に効く** → imitation 系 (case1/case4) の featurizer に
取り込めば、policy が「敵が到達する時点で planet がどれだけ持ちこたえるか」を学習できる可能性が高い。
既存 case1 featurizer は集計のみ (incoming sum / nearest_eta) で、turn-by-turn の予測列がない。

仮説 (plan.md より): **timeline 由来 6 列 (loss_3turn / ttf / min_owned / surplus / fall_predicted / keep_needed)
を case1 featurizer に追加した case5 を BC training すると、case1 比 +5pp 以上の勝率改善が期待できる。**

## 実行ステータス

| 項目 | 状態 |
|------|------|
| plan.md 作成 | ✅ `docs/experiment/imitation/20260503_case5_ship_prediction/plan.md` |
| 既存 case6 ベンチマーク参照 | ✅ vs v4 64%, vs v5 burst-only 59-60% を確認 |
| case6 vs v4 新規 30 ep 実行 | ⏸ 開始したが 90 分超過予測のため cancel (既存 iter1 100 ep が同等情報を持つため冗長) |
| imitation case5 実装 | ❌ Cycle 4 以降 (`bot/pipeline/imitation/case5/` の case1 コピー + featurizer 拡張 + DVC stage 追加) |
| RunPod Step A (infra smoke) | ❌ 未実施 (Cycle 2 と同じく対話 prompt 介入待ち) |
| ローカル対戦 (imitation case5) | ❌ training 完了後 (Cycle 5+) |

## ブロッカー / 引き継ぎ

1. **RunPod Step A は依然 Cycle 3 で打てず**: 対話 prompt (offer 選択) の bypass 不可、ユーザー手元実行待ち。
2. **Cycle 4 のスコープ候補** (優先度順):
   - **A**: Step A smoke を打てる timing なら最優先 (preprocess 検証 ≤$0.2)
   - **B**: imitation case5 の case1 コピー + featurizer 拡張 + unit test (training 不要、ローカルで完結)
   - **C**: Cycle 4 で B 完了 → Cycle 5 で Step A (infra) → Cycle 6 で case5 RunPod training (Step B 級 ~$1.0)

## Cycle 3 で得た主要知見

- **ship-prediction = `simulate_planet_timeline`** という認識が確定
- rulebase での効力データ (vs v4 64% / vs v5 cap=3 で 59.7%) を踏まえると、imitation case5 は
  少なくとも featurizer が壊れていない範囲で **case1 baseline と同等以上の勝率** が下限と予想
- `bot/pipeline/imitation/case5/policy/timeline.py` (新規) として `simulate_planet_timeline` を
  copy する設計は cross-case import 禁止 (`.claude/rules/bot/pipeline.md`) に整合

## Cycle 4 推奨タスク

時間配分案 (1h):

1. (10 min) plan.md を再読、Cycle 3 知見を反映して必要なら微調整
2. (25 min) `cp -r bot/pipeline/imitation/case1 bot/pipeline/imitation/case5` → import path を case5 に書換 → `simulate_planet_timeline` を `case5/policy/timeline.py` にコピー
3. (15 min) featurizer.py に timeline 由来 6 列を追加、`PLANET_FEAT_DIM` を 11→17 に変更
4. (10 min) unit test を 1-2 本 (`tests/pipeline/imitation/case5/test_featurizer_timeline.py`)
5. (残り) `dev/test-bot` 実行、commit, push

これなら Cycle 5 で Step A (infra preprocess smoke) → Cycle 6 で case5 を RunPod 学習に回せる。
