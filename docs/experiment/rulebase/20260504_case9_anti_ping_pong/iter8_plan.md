# rulebase/case9 — anti_ping_pong (iter8 plan)

> 作成日: 2026-05-06
> 関連: `iter1-7_*.md`、特に iter5 analysis (余剰 ship 流用問題)
> スコープ: HARASS の発火閾値引き上げで余剰を capture に回す最小実験

## 仮説 (Hypothesis)

iter5 analysis で「余剰 ship が reinforce/harass の小型輸送に消費されすぎ、
production 増強につながる capture に回らない」と特定。
ACCUMULATE port は失敗 (-7pp) だったが、別アプローチとして **HARASS の発火閾値を上げて
低 production target への発火を抑制** すれば、余剰 ship が capture / reinforce
の本命用途に流れて勝率改善する可能性。

具体的には `HARASS_MIN_TARGET_PRODUCTION: 2 → 3` の 1 行変更:
- 現状: production=2 の敵惑星でも harass 発火
- 変更後: production ≥3 の戦略的 target にしか harass しない

iter7 中盤 (0-120戦 累積 58.3%) で +5pp 帯に乗っていた = 設計に再現性のある
改善要素が眠っている可能性が示唆されており、最小コスト介入で確認する価値あり。

## スコープ (Scope)

**変更ファイル**: `bot/pipeline/rulebase/case9/baseline/core/config.py` 1 行のみ
- `HARASS_MIN_TARGET_PRODUCTION: 2 → 3`

**変更しないファイル**: それ以外すべて (case9 = iter2 + iter6 cache 維持)

## 実装ステップ

1. config.py 1 行変更
2. ruff/mypy/snapshot test 確認 (action 系列が変わる場合は snapshot 更新)
3. 200戦評価: `compare_v4.py -n 100 -p 4 --seed 10000`
4. ETA ~70 min (iter6 cache あり)

## 検証方法

- 対戦相手: baseline_v4
- エピソード: **200戦** (300戦は時間がかかりすぎ、200戦で勝率傾向を見る)
- しきい値:
  - **≥55%**: 採択 (+5pp 達成)
  - **51.5% ≤ x < 55%**: 弱採択、iter9 で 300戦 confirm
  - **< 51.5%**: 棄却

## 想定リスク

- production=2 の敵惑星も時として戦略的 target になりうる (harass で奪うと相手の
  production 8 → 6 で大きい)。引き上げで harass 発火回数が大幅減少 → 勝率低下の可能性
- **case4 base の HARASS_MIN_TARGET_PRODUCTION=2 は tuning 結果**。引き上げは逆方向

## 次の判断

iter8 結果次第:
- 採択 → loop 完了に近づく、iter9 で 300戦 confirm
- 棄却 → iter9 で別軸 (capture mission の score weight 強化、または完全 ACCUMULATE+STAY port)
- いずれにせよ iter9 で結論を出す
