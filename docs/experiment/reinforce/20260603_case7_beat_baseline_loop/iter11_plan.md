# case7 「ルールベースに勝つ」ループ — iter11 PLAN

時刻: 2026-06-03 12:02 (cron tick 15)

## iter10 からの学び
- horizon=500 で terminal 報酬復活 (vs self reward +1.79)、学習信号は正常化。
- だが vs lite は reward -2.0 で**飽和** = 勾配の多様性ゼロ → どの行動が less-bad か
  学べない。lite-stage は eval 性能を劣化 (score 51→0)。
- best.pt は win_rate>=best で保存 → self-play 1.0 に張り付き、eval 無関係な model 保存。

## iter11 方針 (2 つの改善)
1. **saturating な lite を curriculum から外す**: `noop → pool` (self_snapshot のみ)。
   pool は過去自分を f_hard で選び難度を上げる = 報酬が飽和しない範囲で難化。
2. **non-saturated shaping**: `combined` (ship diff coef 0.02 + planet diff coef 0.5)。
   ratio は負け時 Φ:1→0 で -1 飽和するが、diff ベースは絶対量に応じ勾配が残る。
   拡張・増産 (production gap の本質) を連続的に報酬化する狙い。
3. horizon=500 固定、16 iter、ep4。

## 期待 / 判定
- self-play で reward 正の上昇が続けば学習継続の証拠。
- 完走 → 10戦 vs v1。score gap が 0→改善 (model が拡張する) か確認。
  ただし self-play のみなので v1 越えは依然不確実 (transfer 問題)。

## 留意 (iter10 で確定)
- horizon=500 を全 config 既定に (memory `project_reinforce_horizon_terminal_reward_bug`)。
- v1 越えは scale が要る可能性大 (Generals.io 論文)。本 iter は「学習が健全に進む
  設定」を詰める段階。
