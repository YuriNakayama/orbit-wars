# Orbit Wars 模倣学習ベースライン設計メモ

> 作成日: 2026-04-20
> 作業ブランチ: `feature/orbit-wars-imitation`（research リポジトリ側）
> 関連: `20260418_baseline.md`, `20260419_imitation_case1_diagnosis.md`

本文書は、Orbit Wars コンペへ **模倣学習 (Imitation Learning, IL)** を適用する際の
ベースライン設計と、過去 Kaggle Simulation コンペから抽出した適用可能手法をまとめる。

調査ログ原典: `research` リポジトリの
`docs/research/runs/kaggle_orbit_wars/retrieval/20260420_imitation_learning/01〜09`

---

## 目次

1. [結論サマリ](#1-結論サマリ)
2. [なぜ IL か（意思決定ログ）](#2-なぜ-il-か意思決定ログ)
3. [過去コンペ事例カタログ](#3-過去コンペ事例カタログ)
4. [IL 基礎: BC / DAgger / IL→RL](#4-il-基礎-bc--dagger--ilrl)
5. [Phase 1: Semantic Segmentation BC（Day 1–5）](#5-phase-1-semantic-segmentation-bcday-15)
6. [Phase 2: Entity / Autoregressive BC（Day 6–10）](#6-phase-2-entity--autoregressive-bcday-610)
7. [Phase 3: BC + 浅い MCTS（Day 11–14）](#7-phase-3-bc--浅い-mctsday-1114)
8. [Phase 4: IL→MARL（Day 15–21+、任意）](#8-phase-4-ilmarlday-1521任意)
9. [Meta Kaggle Episodes データパイプライン](#9-meta-kaggle-episodes-データパイプライン)
10. [ハイパーパラメータ表](#10-ハイパーパラメータ表)
11. [リスクと緩和](#11-リスクと緩和)
12. [着手順チェックリスト](#12-着手順チェックリスト)
13. [参考文献](#13-参考文献)

---

## 1. 結論サマリ

| フェーズ | 期間 | 手法 | 期待到達点 |
|----------|------|------|-----------|
| Day 1–5 | ベースライン | **Halite IV 型 semantic segmentation BC** (小型 U-Net) | LB 中位 |
| Day 6–10 | 改良 | **Kore 2022 型 entity / autoregressive BC** (Transformer) | LB 上位 30% |
| Day 11–14 | 探索統合 | **BC policy + 浅い PUCT MCTS** (Hungry Geese 3位型) | LB 上位 10–20% |
| Day 15–21+ | 自己対戦 | **AlphaStar 型 SL→MARL** (HandyRL / 小規模 league) | LB 上位 5%（任意） |
| 並行 | 補助 | **DAgger** (強ヒューリスティック expert) | compound error 緩和 |

**Orbit Wars は未知の軌道物理を含むため、RL スクラッチで報酬設計するより上位 bot の意思決定を真似るほうが立ち上げが速い。** IL で「軌道力学の常識」を暗黙学習させ、余裕があれば RL で上乗せする戦略。

---

## 2. なぜ IL か（意思決定ログ）

| 候補 | 立ち上げ時間 | 最終到達性能 | Kaggle 実績 | 判定 |
|------|-------------|------------|------------|------|
| Pure Rule-based | 1 日 | 低〜中 | Halite 1 位等 | ベースラインとして別途保持 |
| Pure RL (PPO scratch) | 2–3 週 | 中〜高 | Lux 1 位等 | **Orbit Wars 初見で報酬設計コスト大** |
| BC from top bots | 3–7 日 | 中 | Lux S3 3 位, Kore 上位 | ★ **まずこれを起点にする** |
| BC + MCTS | 1–2 週 | 中〜高 | Hungry Geese 上位 | 次フェーズ |
| IL→RL hybrid | 3 週+ | 高 | AlphaStar, AlphaGo | 余裕があれば |

### 判断根拠

- **既存ベースライン (`20260418_baseline.md`) に対する上積み**: ルールベースに IL ヘッドを足す形で差分投入しやすい。
- **case1 診断 (`20260419_imitation_case1_diagnosis.md`) の示唆**: 序盤の艦隊配分で損をしており、上位 bot の「先手発射判断」を真似ることで即効性がある。
- **Kaggle のクロック**: LB でランク確認しながら改善を回す前提では、学習ループが短い IL が適合する。

---

## 3. 過去コンペ事例カタログ

### 3.1 Lux AI Season 3（2025, 3位 — IL）

- **コンペ**: Lux AI S3（リソース収集・ユニット制御のマルチエージェント）
- **採用**: **Behavioral Cloning + 補正 RL**
- **観測**: マップを多チャネルテンソル (C, H, W) に変換、味方/敵ユニット、リソース、視界マスク、タイル種別
- **行動**: 各ユニットごとに離散行動（上下左右/掘削/建造等）
- **キー実装**: `(B, n_units, action_dim)` の per-unit head、勝者リプレイのみ学習
- **教訓**: 「マルチユニット同時制御」問題では、per-entity head を持つ IL が BC スクラッチの最短路

→ Orbit Wars の複数惑星同時指令にそのまま写像可能。

### 3.2 Kore 2022（khanhvu207 — Autoregressive Transformer IL）

- **コンペ**: Kore 2022（艦隊計画を文字列で指令）
- **採用**: **ship plan (文字列) を Transformer で autoregressive 生成**
- **モデル**: entity encoder（惑星・艦隊を可変長 token に）+ Transformer decoder
- **出力**: `(src_planet, tgt_planet, fleet_ratio_bin, launch_tick)` の token 列
- **教訓**:
  - マクロ戦略を **文字列（プログラム）として生成** する発想は可変長指令に強い
  - 上位 5 チームのリプレイを混合して overfit 回避

→ Orbit Wars の「複数惑星へ同ターン発射」は token sequence にそのままマップ可能。

### 3.3 Hungry Geese（Maxwell 3 位 — BC + MCTS）

- **コンペ**: Hungry Geese（4 エージェント蛇バトル）
- **採用**: **BC policy を prior として PUCT MCTS**
- **シミュレーション数**: 提出時 10–50 node / ターン（1 秒制約内）
- **Value head**: policy net と dual head
- **重要な工夫**: **Rollout policy にも BC を使う**（ランダム rollout は Orbit 系と相性が悪い）
- **教訓**: BC 単体天井を MCTS で 100–200 ELO 押し上げる王道

→ Orbit Wars の軌道計算は時間方向の深い読みが効く → MCTS との相性良好。

### 3.4 Halite IV（Kha Vo — Semantic Segmentation IL）

- **コンペ**: Halite IV（船・シップヤード制御）
- **採用**: **盤面を (C, H, W) テンソル → U-Net → 各セル per-cell action 予測**
- **モデル**: 小型 U-Net（encoder 4 block, decoder 4 block）
- **出力ヘッド**:
  - action_type per cell: {stay, move, launch}
  - fleet_fraction per cell: scalar (0, 1]
  - target_planet per cell: softmax over planet IDs
- **教訓**: グリッド状況では **semantic segmentation としての IL が最も実装が軽くて速い**

→ Orbit Wars の盤面を 2D 化すれば、Day 1 に出せるミニマル BC になる（**推奨 Day 1 ベース**）。

### 3.5 AlphaStar（SL→MARL League）

- **論文**: Vinyals et al., Nature 2019
- **採用**: **Supervised Pretraining (IL) → Population-based Multi-Agent RL**
- **SL データ**: Blizzard 提供 971,000 リプレイ（MMR > 3500、上位 22%）
- **SL 単体で上位 16%** に到達 → RL で更に上乗せ
- **League 構成**: Main Agent + Main Exploiter + League Exploiter の 3 系統
- **教訓**:
  - **「SL で強い初期値を作ってから RL」** が Kaggle の限られた時間でも妥当
  - Entity Transformer は可変ユニット問題をクリーンに解く
  - League は最低 2 系統（Main + Exploiter）で non-transitivity を避ける

→ Orbit Wars でのフル league は不要だが、2 系統 self-play 程度は現実線。

### 3.6 AlphaGo（SL Policy Network）

- **論文**: Silver et al., Nature 2016
- **採用**: **KGS 3000 万手 → 13 層 CNN → 次手確率（accuracy 57%）**
- **構成**: SL policy（prior）+ RL policy（rollout）+ Value net + MCTS
- **教訓**: **IL→RL→MCTS の 3 段重ねが極めて有効**
  - Kaggle 圧縮版: LB 上位 bot 数万〜数十万局面 → PPO → dual head policy/value → 浅い PUCT

---

## 4. IL 基礎: BC / DAgger / IL→RL

### 4.1 Behavioral Cloning (BC)

```
L = Σ_{(s, a) ∈ D_expert} -log π_θ(a | s)
```

- 単純な supervised learning。(state, action) ペアから policy を学習
- **欠点**: 学習時の state 分布は expert のもの、推論時は自 policy のもの → **covariate shift**（compound error が T² で累積）

### 4.2 DAgger (Ross+ 2011)

```
D ← {}
π_1 ← BC(expert trajectories)
for i = 1..N:
    traj ← rollout(π_i)                      # 自 policy で rollout
    labels ← expert(traj.states)             # expert に正解問い合わせ
    D ← D ∪ {(s, expert_a) for s in traj}
    π_{i+1} ← supervised_train(D)
```

- **保証**: 誤差累積が O(T²ε) → O(Tε) に改善
- **Kaggle 流 expert 代替**:
  1. 強ヒューリスティック bot（軌道 intercept 計算つきルールベース）
  2. 手元の BC policy + 長時間 MCTS（計算時間 ≫ 提出時）
  3. LB 上位 bot リプレイからの k-NN（類似状態の expert action を再利用）
- iter 数は 3–5 が現実的（rollout + 再学習コスト）

### 4.3 IL→RL

- BC 初期化 → PPO / V-trace / REINFORCE で self-play
- 初期 rating が BC で十分高ければ RL で +100–300 ELO が現実的
- Kaggle では HandyRL（DeNA 提供の分散 RL）や stable-baselines3 が軽量

---

## 5. Phase 1: Semantic Segmentation BC（Day 1–5）

**参考**: cluster 04（Halite IV Kha Vo）

### 5.1 観測テンソル `(B, C, H, W)`

| ch | 内容 |
|----|------|
| 0 | 自艦隊量（正規化） |
| 1 | 敵艦隊量 |
| 2 | 惑星中心位置マスク |
| 3 | 惑星 ID（embed 入力） |
| 4 | 所有者（自 / 敵 / 中立） |
| 5 | 残りターン比率 |
| 6–9 | 過去 4 ターンの敵行動履歴 |
| 10–13 | 軌道予測: 次 N ターンの惑星位置 |
| 14–17 | その他（太陽引力場、距離マップ 等） |

- チャネル数 C = 14–18 目安
- H, W はコンペ盤面サイズに合わせる（典型 32×32〜64×64）

### 5.2 モデル: 小型 U-Net

- Encoder 4 block（ch: 32 → 64 → 128 → 256）
- Decoder 4 block（skip connection あり）
- Total params 〜 10M（Kaggle 推論時間に収まる上限）

### 5.3 出力ヘッド

- `action_type` per cell: `{stay, launch}` の 2 クラス softmax
- `fleet_fraction` per cell: scalar in (0, 1]（MSE or binned softmax）
- `target_planet` per cell: softmax over planet IDs

### 5.4 損失

```
L = CE(action_type)
  + MSE(fleet_fraction * launch_mask)
  + CE(target_planet * launch_mask)
```

`launch_mask` は action_type == launch のセルだけを通すマスク。

### 5.5 データ量

- LB 上位 50 bot × 20 エピソード ≒ 1000 ep × 500 step = **500k サンプル**
- 対称性 (左右反転 / 180° 回転) で ×2〜×4 に augment（盤面性質次第）

### 5.6 PyTorch スケルトン

```python
import torch, torch.nn as nn, torch.nn.functional as F

class OrbitUNet(nn.Module):
    def __init__(self, in_ch=18, n_planets=16, base=32):
        super().__init__()
        self.enc1 = self._block(in_ch, base)
        self.enc2 = self._block(base, base*2)
        self.enc3 = self._block(base*2, base*4)
        self.enc4 = self._block(base*4, base*8)
        self.dec3 = self._block(base*8 + base*4, base*4)
        self.dec2 = self._block(base*4 + base*2, base*2)
        self.dec1 = self._block(base*2 + base,   base)
        self.head_action  = nn.Conv2d(base, 2, 1)
        self.head_frac    = nn.Conv2d(base, 1, 1)
        self.head_target  = nn.Conv2d(base, n_planets, 1)

    def _block(self, ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GELU(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([F.interpolate(e4, scale_factor=2), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return {
            "action": self.head_action(d1),
            "frac":   torch.sigmoid(self.head_frac(d1)),
            "target": self.head_target(d1),
        }

def loss_fn(out, gt):
    la = F.cross_entropy(out["action"], gt["action"])
    mask = (gt["action"] == 1).float()
    lf = F.mse_loss(out["frac"][:, 0] * mask, gt["frac"] * mask)
    lt = F.cross_entropy(out["target"], gt["target"], reduction="none")
    lt = (lt * mask).sum() / mask.sum().clamp_min(1)
    return la + lf + lt
```

### 5.7 成功判定

- 訓練 loss が単調減少し validation accuracy（action_type）が 60%+
- LB 提出で既存ルールベースに同等以上 → Phase 2 へ

---

## 6. Phase 2: Entity / Autoregressive BC（Day 6–10）

**参考**: cluster 02（Kore 2022 khanhvu207）

### 6.1 設計

- 惑星・艦隊を **entity list**（可変長）として扱い Transformer encoder に投入
- Action を `(src_planet, tgt_planet, fleet_ratio_bin, launch_tick)` の **token sequence** として autoregressive 生成
- 1 ターン中の全発射指令を 1 sequence で出力 → **同時最適化**（Phase 1 の per-cell 独立決定より強い）

### 6.2 モデル

- Entity encoder: 8-head Transformer × 4 layer、d_model=128
- Decoder: causal masked Transformer × 4 layer
- Pointer network で src/tgt 惑星選択

### 6.3 学習

- teacher forcing で token 列の CE
- fleet_ratio は 10 bin の softmax に離散化

### 6.4 いつ Phase 2 を選ぶか

- Phase 1 の per-cell 独立決定では「協調攻撃」「供給線」が学習できない兆候（LB 上位 bot のリプレイで 2 惑星同時発射パターンが多い）

---

## 7. Phase 3: BC + 浅い MCTS（Day 11–14）

**参考**: cluster 03（Hungry Geese Maxwell 3 位）

### 7.1 PUCT MCTS 擬似コード

```python
def puct_search(root_state, policy_net, value_net, n_sim=30, c_puct=1.5):
    root = Node(root_state, prior=policy_net(root_state))
    for _ in range(n_sim):
        node, path = root, []
        while node.expanded and not node.terminal:
            a = max(node.children,
                    key=lambda c: c.Q + c_puct * c.P *
                                  (node.N ** 0.5) / (1 + c.N))
            path.append((node, a))
            node = node.children[a]
        if not node.terminal:
            v = value_net(node.state)
            node.expand(policy_net(node.state))
        else:
            v = node.terminal_value
        for parent, a in reversed(path):
            child = parent.children[a]
            child.N += 1
            child.Q += (v - child.Q) / child.N
            v = -v  # zero-sum で符号反転
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]
```

### 7.2 パラメータ

- 提出時 1 秒/ターン制約 → n_sim = **10–50**
- c_puct = 1.5 付近から調整
- rollout policy に **BC policy を再利用**（ランダム rollout は軌道系と致命的に合わない）

### 7.3 期待上昇

- Hungry Geese の事例では BC 単体から +100–200 ELO
- Orbit Wars は軌道予測の長手読みが効くためさらに上積みの余地

---

## 8. Phase 4: IL→MARL（Day 15–21+、任意）

**参考**: cluster 05 (AlphaStar), cluster 06 (AlphaGo)

### 8.1 構成

- HandyRL で BC init → PPO or V-trace self-play
- **最小 league**: Main × 1 + Exploiter × 1 の 2 系統
- Value head は BC policy net と共有(dual head)

### 8.2 期待効果

- 自分の BC rating が十分高い場合、RL で +100–300 ELO
- league を持たないと self-play の non-transitivity で ELO 頭打ち or 崩壊

### 8.3 時間配分の目安

- BC checkpoint からの fine-tune 3–7 日
- CPU では PPO 1 iter 数分〜十数分。GPU があれば大幅短縮

---

## 9. Meta Kaggle Episodes データパイプライン

**参考**: cluster 08

### 9.1 データ源の 3 レイヤ

| レイヤ | 取得手段 | 内容 |
|-------|---------|------|
| A. Episode metadata | Meta Kaggle CSV (`Episodes.csv`, `EpisodeAgents.csv`) | エピソード ID、参加チーム、終了時刻 |
| B. Episode replay | `DownloadEpisode` API | 各ターンの観測・行動 JSON |
| C. Submission code | Meta Kaggle Code / 公開 kernel | 解法参考 |

### 9.2 上位 bot エピソード抽出

```python
import pandas as pd

episodes = pd.read_csv("Episodes.csv")
agents   = pd.read_csv("EpisodeAgents.csv")

orbit_wars_id = <competition_id>
target = episodes[episodes.CompetitionId == orbit_wars_id]

agents = agents[agents.EpisodeId.isin(target.Id)]
top    = agents.nlargest(100, "UpdatedScore").SubmissionId.unique()
ep_ids = agents[agents.SubmissionId.isin(top)].EpisodeId.unique()
```

### 9.3 リプレイ本体のダウンロード

```python
import requests, json, time

URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/DownloadEpisode"
for eid in ep_ids:
    r = requests.post(URL, json={"EpisodeId": int(eid)})
    json.dump(r.json(), open(f"replays/{eid}.json", "w"))
    time.sleep(1.0)  # rate limit 配慮
```

### 9.4 観測/行動対への展開

```python
from kaggle_environments import make

env = make("orbit_wars")
env.steps = episode_json["steps"]

samples = []
for step_idx, step in enumerate(env.steps):
    for agent_idx, agent_state in enumerate(step):
        a = agent_state.action
        if a is None: continue
        samples.append({
            "obs":       tensorize(agent_state.observation),
            "action":    encode_action(a),
            "agent_idx": agent_idx,
            "step":      step_idx,
        })
```

### 9.5 フィルタ指針

- 勝者側のみ学習 / 両側使う場合は勝率で sample weight
- bot ごと episode cap（例: 1 bot 50 ep まで）で overfit 回避
- 対称性で ×2〜×4 augment
- 自分の bot の episode は **除外**（team_id マッチで落とす）— 自己教師化で overfit する

### 9.6 保存

- parquet（episode 単位 shard）
- 観測は `(C, H, W)` の numpy / torch tensor で **事前 materialize**（逐次 tensorize は遅い）
- PyTorch `IterableDataset` + shuffle buffer で学習ループへ

---

## 10. ハイパーパラメータ表

| 項目 | Day 1 (U-Net BC) | Day 7 (Transformer BC) | Day 14 (BC+MCTS) |
|------|------------------|------------------------|-------------------|
| optimizer | AdamW | AdamW | — |
| lr | 3e-4 | 2e-4 | — |
| batch | 128 | 64 (seq) | — |
| weight_decay | 1e-4 | 1e-4 | — |
| epoch | 10–20 | 10–20 | — |
| mixed precision | fp16 / bf16 | bf16 | — |
| parameter 数 | 〜10M | 〜20M | — |
| augment | flip / rot | token shuffle (安全な順序のみ) | — |
| MCTS n_sim | — | — | 10–50 / ターン |
| c_puct | — | — | 1.5 |

---

## 11. リスクと緩和

| リスク | 緩和策 |
|-------|--------|
| **covariate shift** | DAgger（強ヒューリスティック expert）、self-play 生成データを混合 |
| **特定 bot overfit** | 上位 N チームを混合、bot 毎 cap を設ける |
| **観測仕様変更（コンペ途中）** | 取得時の env version をメタに残す、差分吸収層 or 再学習 |
| **Kaggle 推論時間制限** | モデル 10M param 以内、batch inference、C++/ONNX export 検討 |
| **IL 天井** | Phase 3 (MCTS) or Phase 4 (RL) へ移行 |
| **自己リプレイ混入** | team_id マッチで排除。Meta Kaggle の user_id も突合 |
| **Rate limit (Kaggle API)** | 分あたり 60 req 程度、指数バックオフ |
| **データ偏り** | 複数 top bot 混合（Kore 2022 top5 混合と同じ発想） |

---

## 12. 着手順チェックリスト

- [ ] Orbit Wars 公式仕様確定後、観測・行動仕様を確認し、Phase 1 モデルの channel / output 仕様を確定
- [ ] `Episodes.csv` / `EpisodeAgents.csv` の日次 pull スクリプト
- [ ] Top-N bot のリプレイ取得 worker（parquet 化）
- [ ] `kaggle-environments` での observation / action tensorizer
- [ ] PyTorch `IterableDataset`（shuffle buffer 付き）
- [ ] Phase 1 U-Net PyTorch 実装（〜100 行）
- [ ] 小リプレイセットで loss 下降確認
- [ ] LB 初サブミット → ルールベース比較
- [ ] 必要なら Phase 2 (Transformer) に移行
- [ ] Phase 3 (MCTS) 移行判断（BC 天井が見えたら）
- [ ] 余裕があれば Phase 4 (RL self-play)

---

## 13. 参考文献

### 論文

- Ross, Gordon, Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning", AISTATS 2011. <https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf>
- Silver et al., "Mastering the game of Go with deep neural networks and tree search", Nature 2016. <https://www.nature.com/articles/nature16961>
- Silver et al., "Mastering the game of Go without human knowledge" (AlphaGo Zero), Nature 2017.
- Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning", Nature 2019. <https://www.nature.com/articles/s41586-019-1724-z>
- DADAgger (2023): <https://arxiv.org/abs/2301.01348>

### Kaggle writeups（要アクセス認証）

- Lux AI Season 3 3rd — Behavioral Cloning
- Kore 2022 — khanhvu207 Autoregressive Transformer
- Hungry Geese — Maxwell 3rd BC+MCTS
- Halite IV — Kha Vo Semantic Segmentation（khavo.ai は一時 404、Wayback 推奨）

### コード / フレームワーク

- Kaggle/kaggle-environments: <https://github.com/Kaggle/kaggle-environments>
- HumanCompatibleAI/imitation (DAgger 実装): <https://imitation.readthedocs.io/en/latest/algorithms/dagger.html>
- DeNA/HandyRL: 分散 RL
- Fkaneko/kaggle_lux_ai: episode 収集スクリプト参考
- RoboEden/Luxai-s2-Baseline: 公式推奨 IL データ生成
- kimbring2/AlphaStar_Implementation: 再現実装

### research リポジトリ側原典

- `docs/research/runs/kaggle_orbit_wars/retrieval/20260420_imitation_learning/index.md`
- 同 `01-lux-s3-3rd-imitation.md`
- 同 `02-kore2022-autoregressive-il.md`
- 同 `03-hungry-geese-bc-mcts.md`
- 同 `04-halite-iv-semantic-segmentation-il.md`
- 同 `05-alphastar-supervised-pretraining.md`
- 同 `06-alphago-sl-policy-network.md`
- 同 `07-dagger-dataset-aggregation.md`
- 同 `08-meta-kaggle-episodes-pipeline.md`
- 同 `09-orbit-wars-il-design-memo.md`

---

## 付録: 用語

| 用語 | 意味 |
|------|------|
| BC | Behavioral Cloning、(s,a) ペアの supervised learning |
| DAgger | Dataset Aggregation、自 policy rollout + expert ラベルで covariate shift 補正 |
| PUCT | Predictor + UCT、AlphaGo/AlphaZero の MCTS 選択規則 |
| MARL | Multi-Agent Reinforcement Learning |
| Pointer network | 可変長候補集合から 1 つを選ぶ softmax 層（ユニット選択等） |
| League training | 複数 agent を並走させ self-play の non-transitivity を回避する訓練法 |
