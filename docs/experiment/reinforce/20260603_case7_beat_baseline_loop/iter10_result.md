# case7 「ルールベースに勝つ」ループ — iter10 RESULT (★BUG 発見)

時刻: 2026-06-03 11:01 (cron tick 13 / reward 分析の深掘りから)

## reward 分析 → 異常検出 → ★根本バグ特定
reward 推移分析で「**vs noop でも win=0.500 / reward≈0**」という異常に気づいた。
noop = 何もしない相手なのに勝率 0.5 は不自然 (vs random は 6/6, score 7242:242 で圧勝)。

### 原因: horizon 不足で terminal 報酬が一度も発火していなかった
- Orbit Wars のゲームは **~497-500 turn 続く** (both-noop でも term=turn 497 を実測)。
- terminal_reward は `where(term, sign(r_self-r_opp), 0)` = **ゲーム終了時のみ ±1**。
- **本ループの学習 config は全て horizon=200** (`local_combo_20min.yaml` 由来、速度優先で
  私が設定)。→ **200 turn で打ち切り = ゲームは絶対に終わらない = term 常に False**
  → **terminal_reward が毎 step 0** → episode_outcome = shaping のみ ≈ 0
  → win_rate = mean(outcome>0) ≈ 0.5 (shaping noise のコイン投げ)。
- つまり **モデルは「勝つ」報酬を一度も受け取っていなかった**。PPO は shaping だけで
  学習し、実際の勝敗を学べない → eval (horizon 500 フルゲーム) で 0/10。

### 証拠
- production config (`kaggle_jax_train.yaml`) は **horizon=500**、case5/6 も 500。
  → 本来 500 が正。私の loop config の horizon=200 が **self-inflicted bug**。
- vs random は 6/6 圧勝 = モデルは行動できる。0/10 は「勝ち報酬を見ずに学習した」結果。

## ★結論の訂正
これまでの「ローカル小規模 RL では構造的に勝てない」は **誤り (の可能性大)**。
真因は **horizon=200 で terminal 報酬が消えていた config バグ**。
horizon=500 (ゲームが終わる長さ) で学習し直せば、勝敗報酬が入り学習が機能する見込み。

## NEXT ACTION (即実行)
1. **horizon=500 で再学習** (BC warmstart + ratio shaping + curriculum)。
   ゲームが終わるので terminal ±1 が発火 → 正しい reward 信号。
   episodes は重さを抑えるため 4-6、iters 12-14。
2. 完走 → 10戦 vs v1。vs noop の win が 1.0 に近づくか (報酬が正しく入った証拠) も確認。
3. これで動けばループの本来の目的に復帰。動かなければ scale 議論へ。

## ★fix の即時検証 (11:09)
- 直接テスト: 未学習 model を horizon=500 vs noop で rollout →
  **valid steps=498 (ゲーム終了!)、episode_outcomes=[-0.98,+1.47,+1.67,+1.88]、
  win_rate=0.75**。horizon=200 では terminal が消えていたのが、500 で**復活**。
- iter10 学習 trace: **iter2 (self_snapshot) で win=1.000 / reward=+0.82** が出た。
  horizon=200 時代の全 run は vs self でも reward ~0 だった → **明確な改善**。
  terminal ±1 報酬が入り、PPO が実際の勝敗を学習し始めた証拠。
- → **「ローカル RL で勝てない」は horizon=200 config バグが主因だった可能性が濃厚**。
  iter10 完走 → 10戦 vs v1 で本来の性能を再評価。


## ★iter10 最終測定 (11:32)
| model | vs baseline_v1 | score gap |
|---|---|---|
| iter10 (horizon=500 fix) | **0/10** | **0 vs 242** (悪化) |

- horizon 修正で **学習信号は復活** (vs self reward +1.79, win 1.0 — h200 時代の ~0 から激変)。
  → **「学習できる状態」には戻った。horizon=200 は実在の致命的 config バグだった**。
- しかし vs v1 は依然 0/10。さらに **score が 51→0 に悪化**:
  lite-stage (iter 8-11) で reward -2.0 が飽和 → 勾配が「とにかく発射しない」方向へ
  押し、best.pt は self-play 最適 (win 1.0) で保存されたため vs v1 で最弱に。
- = horizon バグ修正は**正しく重要**だが、それだけでは v1 に勝てない。lite/v1 の壁は本物。
  saturated -2.0 の lite-stage はむしろ eval 性能を劣化させる (有害)。

## 確定した二層構造
1. **config バグ層 (修正済)**: horizon=200 で terminal 報酬消失 → 13 tick の reward~0 の主因。
   horizon=500 で解消、self-play 学習は正常化 (+1.79)。
2. **本質的な壁 (残存)**: それでも lite/v1 には勝てない (reward -2.0 飽和)。
   self-play で強くなっても本物 v1 に transfer しない = memory の train/eval ギャップ。
   research (Generals.io) の通り、ここを越えるには scale が要る。

## NEXT
- horizon=500 を全 loop config の既定にする (バグ修正の定着)。
- lite-stage の saturated -2.0 は有害 → curriculum を self_snapshot 中心にし、
  lite は最後に少量 or 外す。self-play で reward 正を伸ばす方向が iter で効いている。
- self-play のみで長く回し (horizon=500)、best を self でなく「vs lite 改善時」に保存する
  ロジックへ変更も検討 (現状 best=win_rate で self 1.0 に張り付く問題)。
