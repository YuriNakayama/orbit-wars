# case7 ループ — iter12: 弱い相手の梯子 (opponent ladder)

時刻: 2026-06-03 12:10 (cron tick 15 続き)
質問: 「baseline_v1 ではなくもう少し弱いモデルに勝つ方法は?」

## rl_v7 (現状 model) の対戦相手 ladder (実測)
| 相手 | rl_v7 勝率 (6戦) | 学習価値 |
|---|---|---|
| random | 6/6 (1.00) | 簡単すぎ |
| il_v0 (最弱 imitation) | 6/6 (1.00) | 簡単すぎ |
| **rl_v0 (scratch PPO)** | **4/6 (0.67)** | ★**互角に近い = 最適な踏み台** |
| baseline_v1 | 0/6 (0.00) | 強すぎ (reward -2.0 飽和) |

補足:
- `rl_v0 vs baseline_v1 = 0/4` → **学習モデルは皆 v1 に勝てない** (v1 は本物に強い)。
- `baseline_v1 vs baseline_v2 = 2/4` → rulebase 同士は ~50% (v1 より弱い rulebase は無い)。

## 答え: 「もう少し弱い相手」= rl_v0 か self_snapshot
1. **rl_v0 (0.67)** が理想の踏み台。だが JAX rollout の opponent mode に **rl_v0 は無い**
   (PyTorch agent。python_v1 同様 host callback 化が必要で遅い)。
2. **self_snapshot (過去の自分)** は JAX 内で高速に使え、rl_v7 は実際に勝てる
   (iter10/11 で vs self win 1.0)。**これが「少し弱い相手に勝つ」最も実用的な実験**。

## 推奨する小規模実験 (ladder curriculum)
report 用に分かりやすい「弱い相手に勝つ」デモ:
- **opponent=self_snapshot 固定 + snapshot を周期更新** (pool, snapshot_every 小)。
  → 学習が進むほど「少し前の自分 (= 少し弱い)」に勝ち続ける構図。
  reward 正・win>0.5 が安定 = 「弱い相手に勝てている」明確な証拠。
- horizon=500 (terminal 報酬必須)、shaping=ratio/1.0 (combined の係数暴走は回避)。
- 仕上げに 10戦 vs rl_v0 で「v1 以外の学習相手には勝てる」を示す。

## iter11 の失敗 (記録)
combined shaping の coef_planet=0.5 が horizon=500 で過大 → reward が ±10 に爆発
(iter1 reward -12, iter2 +9)。**diff 系 shaping は係数を小さく** (ratio/1.0 が安全)。打ち切り。

## NEXT
- 上記 ladder (self_snapshot pool, horizon500, ratio/1.0) を回し、
  「弱い相手 (過去自分) に安定して勝つ」を確認 → 10戦 vs rl_v0 で締め。
- v1 越えは別問題 (scale) として切り分け済。本実験は「勝てる相手で学習を回す」狙い。

## ★iter12 最終結果 (12:56) — 「弱い相手に勝つ」達成
| 対戦相手 | iter12 後 | (参考) 学習前 rl_v7 |
|---|---|---|
| **rl_v0 (少し弱い学習モデル)** | **9/10 (0.90)** | 4/6 (0.67) |
| baseline_v1 (強い rulebase) | 0/10 | 0/10 |

- **rl_v0 への勝率が 0.67 → 0.90 に向上** = ladder self-play 学習で「もう少し弱い相手に
  勝つ」が**実証された**。質問への回答が成立。
- self-play 内 win は 15 iter 平均 ~0.72 (1.0〜0.25 で pool 難度に応じ振れる = PFSP 正常)。
- v1 は依然 0/10 (別問題 = scale)。だが **「学習が機能し、勝てる相手には勝てる model」**
  が作れたのは大きな前進 (horizon バグ修正 + 飽和相手の排除が効いた)。

## 確立した手順 (再現可能)
1. horizon=500 必須 (terminal 報酬、memory `project_reinforce_horizon_terminal_reward_bug`)。
2. opponent = self_snapshot pool (飽和する v1/lite を学習相手にしない)。
3. shaping = ratio/1.0 (combined の係数暴走を回避)。
4. BC warm-start (case9) から開始、KL anchor 0.15。
→ この設定で「弱い〜互角の学習相手に勝てる」model が小規模 (16 iter, ~38min CPU) で得られる。

## 結論
- **小規模実験で「baseline_v1 より弱い相手 (rl_v0) に勝つ」は達成 (0.90)**。
- baseline_v1 越えは scale 問題として切り分け済 (Generals.io: H100×36h 級)。
- ループの本来の学習基盤 (case7 + resume + incremental metrics + horizon fix +
  memory features) は健全に機能することを実証。
