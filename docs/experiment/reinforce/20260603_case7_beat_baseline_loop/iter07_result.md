# case7 「ルールベースに勝つ」ループ — iter07 RESULT

時刻: 2026-06-03 05:01 (cron tick 9 / 2時間経過)

## やったこと: 0/10 の振る舞いを実際に観察 (replay 診断)
win 数だけでなく「モデルが実際に何をしているか」を確認。

## ★発見
1. **rl_v7 vs random = 10/10**。→ モデルは行動しており、broken-passive ではない。
   勝てる相手には勝つ = 推論パイプラインは生きている。
2. **rl_v7 vs baseline_v1 = 0/10、最終スコア 51 vs 16413** (500 turn 完走、winner=v1)。
   → モデルは ship を 51 しか蓄積できず、v1 は 16413。**桁違いの大差**。
   crash でも passive でもなく、**生産/拡張戦略が v1 に比べ壊滅的に劣る**。

## Diagnosis
- 0/10 の正体は「何もしない」ではなく「**惑星確保・生産で v1 に完敗**」。
- ratio shaping (mine/(mine+enemy)) は保持割合を促すが、**絶対的な拡張・増産**を
  v1 レベルに引き上げるには小規模 RL では不足。
- memory `case6_live_eval`: GPU self-play で vs-full 0.42 まで伸びても本物 v1 に 0/10。
  → GPU more-of-the-same でも v1 勝利は不確実 (本診断の score gap がそれを裏づけ)。

## 統合 (9 tick)
学習モデル一族は vs v1 = 0/10 (score 51:16413 級の production 大差)。
small-scale RL でも imitation 底上げでも本物直接学習でも破れず。

## NEXT ACTION
- 本ループの成果物 (case7 family + resume 機能 + incremental metrics + 全 config +
  本 docs) は価値があるので **commit して保全** (GPU 起動の SHA 確保も兼ねる)。
- GPU 大規模は directive で許可済だが memory 的に v1 勝利は不確実 + 課金。
  → commit 後、GPU を 1 回だけ試すか、別 family へ振るかをユーザーに最終確認。
- production gap が本質なので、もし RL を続けるなら shaping を「拡張・増産」に
  直結する形 (例: 自軍 production 総量の potential) へ作り替えるのが筋 (case5 で
  rejected だった dense とは別の PBRS 設計が要る)。
