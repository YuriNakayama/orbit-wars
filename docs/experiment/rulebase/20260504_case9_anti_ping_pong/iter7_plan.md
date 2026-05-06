# rulebase/case9 — anti_ping_pong (iter7 plan)

> 作成日: 2026-05-06
> 関連: `iter1-6_*.md`
> スコープ: iter2 best 設計の 300戦再評価 (loop 完了条件判定用)

## 仮説 (Hypothesis)

iter1-6 の累積で iter2 (200戦 49.5%) が依然 best。Wilson 95% CI は [42.7%, 56.3%]
で +5pp 達成 (≥55%) は信頼区間上限ぎりぎり。**300戦に増やせば CI が ±5pp に縮小**
し、真値が +5pp 達成しているかを判定できる。

case9 = iter2 設計 + iter6 plan_shot cache (動作には影響しない高速化) で
評価所要時間 ~100 分 (iter6 70 min × 1.5)。

## 採否分岐

- **300戦で 55% 以上**: 統計的有意に +5pp 達成 → **iter 採択** + **loop 完了条件達成**
  - memory に成功事例を記録
  - cron loop 停止 (CronDelete 11c931e9)
- **51% ≤ 300戦 < 55%**: iter2 ベースの設計で +5pp は届かないと結論
  - iter8 で別軸 (ACCUMULATE+STAY 同時 port、case4 大改造) に振り切る
- **< 51%**: iter6 cache に何らかの副作用がある可能性、要検証

## スコープ (Scope)

- 変更ファイル: なし (case9 = iter2 + iter6 cache の現状で評価のみ)
- 評価コマンド: `compare_v4.py -n 150 -p 4 --seed 9000` (各 seat 150戦 = 300戦)
- ETA: ~100 分

## 検証方法 (Validation method)

- 対戦相手: baseline_v4 (case4 production)
- エピソード: **300戦**
- 主要メトリクス: vs v4 勝率 + Wilson 95% CI
- 採否しきい値: 上記分岐参照

## 想定リスク

- **iter6 cache の副作用**: 平均試合長が iter2 比 -12 turn 短縮していた点が気になる。300戦で勝率が iter2 (49.5%) を大きく下回るようなら cache の問題が顕在化
- **seed range 依存**: seed 9000- が iter1-6 の seed range と独立、運の影響あり
