# RL pool 構成 & 「強 rule-base に勝率 0」打開策 — リサーチと提案

時刻: 2026-06-04 / web search + 既存 case7 知見の統合

## Part 1: pool 構成 — 文献の結論

### AlphaStar league(PFSP)の 3 役構成
| 役 | 学習相手 | 本 project への対応 |
|---|---|---|
| **Main agent** | league 全体を PFSP(勝てない相手を優先 `(1-p)^p`) | case7 本体(rl_v7) |
| **Main exploiter** | 現 main agent を直接攻撃 → 弱点露呈 | **新設候補**: case7 を倒す専用 agent |
| **League exploiter** | league 全体を PFSP、main の標的にされない | case8(本物 rule)を固定 exploiter として代用 |

**核心**: fictitious self-play 単独では不十分 → league(exploiter 併用)が必要だった
(DeepMind が明言)。past-self だけの pool は forgetting と局所最適に陥る。

### pool に入れるべき相手(文献の合意)
1. **過去自身 snapshot(主軸)**: PFSP で勝率比例サンプル。FIFO でなく **勝率で重み付け**。
2. **exploiter(自身の弱点を突く agent)**: main の穴を露呈させ堅牢化。
3. **「同〜やや上」の相手を中心に**: 強すぎ=勾配消失、弱すぎ=過学習。Elo 的 matchmaking で
   現在実力の近傍を厚くサンプルするのが最適(複数文献が一致)。

### case7 現状とのギャップ
- 現 case7 = past-self FIFO + lite/full 固定混入。**FIFO は勝率非考慮**(AlphaStar は勝率重み)。
- **exploiter(main を直接攻撃する agent)が無い**。lite/full は飽和=「強すぎ」側に寄り、
  「同〜やや上」のゾーンが空。→ **勝率近傍を厚くする matchmaking が欠落**。

## Part 2: 「強 rule-base に勝率 0」打開策 — 文献の処方箋

文献が示す原因: **勝率 0 → terminal 報酬が常に負 → 勾配が「負を薄める」方向にしか働かず、
方策分布が相手分布と乖離しすぎて学習信号が死ぬ**(sparse + distribution mismatch)。

### 処方 A: 逆カリキュラム(Reverse Curriculum, arXiv 1707.05300)★最有力
- **ゴール近傍の state から開始** → 即座に勝ち報酬が出る → 後方へ拡張。
- **本 project への適用**: vs case8 戦で、**中盤の「互角〜やや有利」局面を初期 state に**して
  「勝ち切り」を先に学習 → 徐々に序盤へ後退。0/100 でも「あと少しで勝てた局面」から
  正の信号を取れる。case8 戦の replay から有利局面を抽出して reset state に注入。
- 自動カリキュラム: agent の勝率に応じて開始局面の難度を適応調整。

### 処方 B: 相手ハンディキャップ(handicapping)
- **強相手を一時的に弱体化**(ship 生産 ×0.7、launch 間引き等)→ 勝てる難度から開始 →
  agent 向上に合わせてハンディを 0 へ漸減。「同レベルの相手が共に成長」の近似。
- case8 は rule なので **生産係数 / 行動確率を直接間引ける**(学習相手の難度を連続調整可能)。

### 処方 C: Minimax Exploiter reward(arXiv 2311.17190)★本 project と相性最高
- 報酬に相手価値のペナルティを加算: `R - αγ·max_a Q_opp(s')`。
- 通常は相手 Q を BC で近似要 → **本 project は case8 の評価関数が完全アクセス可能**
  (rule の scoring_jax)。**相手 Q を近似不要で dense 信号**を直接得られる稀有なケース。
- 効果: 23-30% 収束高速化(文献)。sparse な terminal 報酬を dense 化。
- α は 0.01-0.1 から保守的に。

### 処方 D: 勝率重み PFSP + matchmaking
- past-self pool を FIFO → **勝率重み**(`(1-win)^p`)に。「同〜やや上」を厚くサンプル。
- exploiter 混入は **cap で抑制**(既存方針)、強すぎゾーンを薄く。

### 処方 E: 報酬非対称(asymmetric reward)
- 非対称ゲームで弱側が正報酬を取れない問題への定石: 弱側(=agent)の勝ち報酬を増幅、
  負けペナルティを減衰し、初期の正信号を確保。terminal 飽和(-2.0)を緩和。

## Part 3: 本 project への推奨(優先順)

| 優先 | 施策 | 根拠 | コスト | 本 project の利点 |
|---|---|---|---|---|
| **1** | **Minimax reward(処方C)** | case8 scoring 完全アクセス | 低(1 eval/step) | BC 近似不要、dense 化、in-JAX |
| **2** | **逆カリキュラム(処方A)** | sparse の王道、0/100 でも信号 | 中(reset state 注入) | case8 replay から有利局面抽出可 |
| **3** | **勝率重み PFSP + cap(処方D)** | AlphaStar 準拠、FIFO 改善 | 低(selector 改修) | 既存 selector に勝率重み追加 |
| **4** | **ハンディキャップ漸減(処方B)** | 同レベル相手の近似 | 低(case8 係数) | rule なので難度連続調整可 |
| 5 | 非対称報酬(処方E) | 飽和緩和 | 低 | terminal 符号の重み調整 |
| 補 | main exploiter 新設 | league 完全化 | 高(別 agent 学習) | 小規模では過剰、後回し |

**核心メッセージ**: 本 project は case8 の **rule 評価関数が完全に開いている** 稀有な状況。
文献で「BC で近似が必要」とされる Minimax reward と handicapping を **近似なしで直接適用** できる。
これが「強 rule に 0/100」を破る最大の武器。逆カリキュラムと併用すれば sparse 問題に二重対処。

## Sources
- [AlphaStar (DeepMind blog)](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
- [SCC: Efficient DRL Agent for StarCraft II (PFSP 詳細)](https://arxiv.org/pdf/2012.13169)
- [Reverse Curriculum Generation for RL](https://arxiv.org/pdf/1707.05300)
- [Minimax Exploiter: Data Efficient Competitive Self-Play](https://arxiv.org/html/2311.17190)
- [Reverse Curriculum (abs)](https://arxiv.org/abs/1707.05300)
- [Self-Play (HuggingFace DRL course, handicap 解説)](https://huggingface.co/learn/deep-rl-course/en/unit7/self-play)
- [Automated Curriculum by Rewarding Rare Events](https://arxiv.org/abs/1803.07131)
- [Robust Opponent-Aware League Training for SC2 (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf)
