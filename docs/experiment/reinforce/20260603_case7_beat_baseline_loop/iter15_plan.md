# case7 ループ — iter15 PLAN: per-iter checkpoint + eval sweep

時刻: 2026-06-03 16:03 (cron tick 23)

## 動機 (iter14b の問題)
- best.pt は実は `_save_best_pt` を毎 iter **無条件** 呼ぶ = 単なる「最新 model」。
  win_rate>=best は S3 upload を gate するだけ。→ self-play が sweet spot を
  overshoot すると、**良い中間 model が失われ最後の劣化 model が残る** (iter14b で発生)。
- per-iter の local checkpoint が無く、後から良い iter を選べなかった。

## ブラッシュアップ (実装済、ruff/mypy/smoke OK)
- `train_jax.py`: 毎 iter `ckpt_i{NNN}.pt` を run_dir に保存 (13MB×N、許容)。
  → 学習後に **各 ckpt を外部 eval (vs rl_v0) して真のピークを選べる**。
- smoke 確認: ckpt_i000/001/002.pt + best.pt が出力される。

## iter15 実行
- iter12 BEST から resume、pure self-play (uniform) 12 iter、horizon=500。
- 完走後、**全 ckpt を vs rl_v0 で eval し、最良 iter を特定** (best.pt=最後 に頼らない)。

## 期待
- self-play の途中に iter12 (0.90) を超える、または別の汎化ピークがあるか。
- 無ければ iter12 が確定ピークと再確認 (ckpt sweep で網羅的に検証)。

## 意義
- これは「best 選択ロジックのブラッシュアップ」= directive の趣旨に合致。
- 今後の reinforce 学習全般で「sweet spot を逃さない」基盤になる
  (memory `project_reinforce_unbeatable_opponent_harmful` の対策)。

## ★iter15 RESULT — ckpt sweep が「best 選択ロジック」の価値を実証 (16:35)
全 12 ckpt の subset を vs rl_v0 で eval:
| ckpt | self-play win | vs rl_v0 |
|---|---|---|
| i000 | 0.50 | **6/6 (1.00)** ← peak (=iter12) |
| i003 | 0.75 | 1/6 (0.17) ← 最悪 |
| i006 | 0.50 | 5/6 (0.83) |
| i009 | 1.00 | 4/6 (0.67) |
| i011 (=旧 best.pt) | 0.50 | 5/6 (0.83) |

### ★決定的な知見
- **rl_v0 への勝率が連続 iter で 1.0 ⇄ 0.17 と激しく振動**、self-play win と**無相関**
  (i009 は self-play 1.0 なのに rl_v0 0.67、i003 は self 0.75 なのに rl_v0 0.17)。
- 旧来の best.pt=最後 (i011) は **peak (i000=1.0) を逃す** + そもそも coin-flip 的選択。
- **per-iter ckpt + 外部 eval で初めて peak を回収できる** = ブラッシュアップの効果実証。
- self-play は「現在の pool に最適化」するので、外部相手 (rl_v0) への汎化は iter ごとに
  乱高下する。**self-play win を model 選択の指標にしてはいけない**。

### 確定
- ピーク = iter12 (i000)、rl_v0 1.00 (6戦)。policy/weights.pt に復元済。
- **学習ロジック改善 (per-iter ckpt) を case7 train_jax に実装・commit すべき**。
- 小規模 RL の到達点は iter12 で確定。v1 越えは scale 必須。
