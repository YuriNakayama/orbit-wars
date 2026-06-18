# モデルサイズは strict 攻略に十分か — 分析 (2026-06-16)

> 問い: 現行モデル (hidden=192, L=4, heads=8, ind=24) は strict を倒すのに十分な容量か。

## 現行モデルの規模 (実測)

| 構成 | params |
|---|---|
| **現行 hidden=192 L=4 heads=8 ind=24** | **3.15M** |
| hidden=256 L=4 | 5.57M |
| hidden=256 L=6 ind=32 | 7.71M |
| hidden=384 L=6 heads=12 | 17.26M |
| hidden=128 L=3 (縮小) | 1.14M |

Set Transformer (ISAB×4) backbone + per-planet pointer head + value head。

## 決定的証拠: capacity は十分 (size は bottleneck でない)

**同一の 3.15M モデルが full held-out 0.83 を学習できている**:

| run (全て3.15M) | full held-out max | strict_v1 max |
|---|---|---|
| ladder11 | 0.812 | 0.016 |
| ladder18 | **0.828** | 0.016 |

full (baseline_jax_full = movement detector) は決して自明でない強相手で、それに 0.83 勝てる
= **3.15M は複雑な戦略を学習する容量を明確に持つ**。strict が 0% なのは「モデルが小さくて
strict戦略を表現できない」のではなく、**学習信号 (reward/exploration) の問題** (rl_failure_
rootcause.md の degenerate-batch 自壊)。同一容量で一方 (full) は学べ他方 (strict) は学べない
のだから、差は capacity でなく signal にある。

## 文献の裏付け

- 「**bottleneck は model capacity でなく sampling/exploration 戦略**」。ゲームによっては
  NN はpolicyを fit するのに十分で、律速は探索側。
- 大きいNN (256/384 neuron) は早期は良いが**後期の収束が遅い**ケースあり (128が最高return)。
  → むやみな増量は逆効果になりうる。
- scaling は「単に param を増やす」より bottleneck に構造を入れる (MoE 等) 方が効く。
  → 本件は scaling 問題でない。

([Mind the GAP, Pixel RL scaling](https://arxiv.org/html/2505.17749),
[Capacity Loss in RL](https://arxiv.org/pdf/2204.09560),
[Detecting Bottlenecks in DRL](https://rileyse.org/2021/09/16/rl_bottlenecks/),
[microRTS winner DoubleCone ResNet](https://www.themoonlight.io/en/review/a-competition-winning-deep-reinforcement-learning-agent-in-microrts))

## 競合比較

microRTS優勝 (RAISocketAI) は DoubleCone ResNet+SE の conv net だが、**grid (GridNet) action
空間**でユニット数が多い RTS 用。Orbit Wars の per-planet pointer (planet数~12-48) は遥かに
小さい意思決定空間で、3.15M Set Transformer は規模的に妥当。優勝の鍵は size でなく
「invalid action masking + 反復fine-tune」だった (competition_solutions_research.md)。

## 結論

- **モデルサイズ 3.15M は strict 攻略に十分** — 同一モデルが full 0.83 を学習できる事実が
  capacity 十分の直接証拠。文献も「bottleneck は capacity でなく exploration」と一致。
- **増量は推奨しない**: 後期収束が遅くなるリスク、strict の signal 問題は解けない、
  rollout/compile コスト増。
- 真の bottleneck は rl_failure_rootcause.md の通り **degenerate-batch 自壊 + 探索/信号**。
  → 次手は size 変更でなく **RL健全性修正 (A: degenerate guard / B: adv-std下限 /
  C: no_op_bias↓)**。size は据え置き。

## 補足 (将来、capacity を疑う場合の最小実験)
もし RL健全性修正後に「容量不足」を疑う根拠が出たら、hidden=192→256 (3.15→5.57M) の
単一A/B を 1 回だけ行えば十分 (384は過大・収束遅延リスク)。ただし現時点で根拠なし。
