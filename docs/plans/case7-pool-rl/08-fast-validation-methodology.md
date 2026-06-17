# RL 施策の高速検証法 (20分〜2h で有効性を判定する) — リサーチと提案

時刻: 2026-06-05 / web search + 既存 case7 知見の統合

## 課題
本格 RL 学習は 10h+/$ がかかる。施策(Minimax reward / 逆カリキュラム / pool 構成等)が
**有効かどうかを 20分〜2h で判定**し、無効な施策に長時間を費やさない仕組みを作る。

## 文献の中核原則 (Andy Jones "Debugging RL" 他)
> 「テスト実行は『修正にかかる時間=数秒』で終わるべき。フルタスクでデバッグするな。」

検証は **段階のピラミッド**。下ほど速く・安く、上ほど遅く・高価。各段を通過した施策だけ上へ。

```
        ┌─────────────────────────────┐ 採否確定
  L4    │ 本番 GPU (10h+, $)            │ ← L0-3 通過後のみ
        ├─────────────────────────────┤
  L3    │ 短縮代理学習 (20min-2h, CPU)  │ ← 本 plan の主戦場
        ├─────────────────────────────┤
  L2    │ probe 環境 (秒〜分)           │ 施策ロジックの正しさ
        ├─────────────────────────────┤
  L1    │ single-batch overfit (秒)    │ 勾配が流れるか
        ├─────────────────────────────┤
  L0    │ smoke (秒)                   │ 落ちずに 1 経路通るか
        └─────────────────────────────┘
```

## L0-L2: 秒〜分で「実装が壊れていない」を保証

### L0 smoke (既存、秒)
- 2 iter × ep=2 × horizon=500 で例外なく best.pt/metrics 出力。**実装バグの即時検出**。

### L1 single-batch overfit (秒) ★施策検証の最重要 sanity
- **1 batch の transition を繰り返し学習 → loss が急減すれば network/optimizer は学習可能**。
- 新報酬項(Minimax 等)を入れた直後にこれが壊れたら、施策の実装バグ確定。**学習を回す前に弾く**。

### L2 probe 環境 (秒〜分) — 施策ごとに「予測可能な正解」を用意
文献の probe (1 action/1 step 等) を **施策に合わせて自作**:
| 施策 | probe | 期待 (数秒で収束) |
|---|---|---|
| Minimax reward | 相手 Q が明白に高い手 vs 低い手の 2 択 1 step | 低 Q 手を選ぶ方策に収束 |
| 逆カリキュラム | 勝利寸前 state から開始 | 1 手で勝ち、reward≈+1 |
| 勝率重み PFSP | 勝率既知の dummy entry 集合 | 低勝率 entry の選択率が理論値一致 |
| shaping (既存) | mine/enemy 比が単調増える state 列 | potential 差分が符号一致 |

**probe は学習の前に施策ロジックの正しさを隔離検証**。フル学習で「効かない」のが
バグなのか施策無効なのか切り分け不能になるのを防ぐ。

## L3: 20分〜2h で「施策が有効か」を判定 — 実験規模の縮小設計

### 縮小の原則: 「ノイズを潰す軸」は残し、「計算量の軸」を削る
文献 (Andy Jones): **large batch + small network で早期にノイズを抑制**。
本 project の縮小レバー (10h → 20min-2h):

| 軸 | フル | 縮小 | 短縮率 | 有効性検証への影響 |
|---|---|---|---|---|
| iterations | 200+ | **20-40** | 5-10× | trend(傾き)は早期に出る。plateau 値は犠牲 |
| horizon | 500 | **500 維持** | ×1 | ★削るな (terminal 報酬消失バグ、memory) |
| episodes/iter | 多 | **8-16** | — | batch を確保しノイズ抑制 (削りすぎ厳禁) |
| network | full | full 維持 | ×1 | 施策効果は構造依存、変えると比較不能 |
| 評価戦数 | 300 | **paired 30-60** | 5-10× | ↓ paired seed で担保 |
| opponent | 重 python_v* | **軽 in-JAX (case8/self)** | 数× | host hop 回避 (memory) |

→ **iterations と評価戦数を削り、horizon/network/batch は守る**のが要。
施策の「効く/効かない」は **学習曲線の傾き (trend)** に早期に出る (文献: "how fast it learns"
と "where it plateaus" は分離可能、傾きは速く判明)。plateau の絶対値は L4 へ委譲。

### ★paired-seed A/B 評価 (本 project の最大の武器)
- 本 project の既知の痛点: **n<300 評価は信頼不可、勝率が 1.0⇄0.17 振動** (memory)。
- 文献 (Paired Seed Evaluation): **同一 seed 集合で両 variant を評価**すると seed 間相関
  0.68-0.99 → **分散が劇的に減り、少ない戦数で有意差検出**可能。
- **recipe**:
  1. baseline と施策ありを **同じ初期 seed 集合**(同じ初期局面)で対戦させる。
  2. 各 seed で `Δ = 施策ありの結果 − baseline の結果` を取る (common random numbers)。
  3. Δ の符号が揃えば、絶対勝率がノイジーでも **差は有意**。
  4. paired t-test / 符号検定で 30-60 戦でも判定可。
- → 「300戦回さないと分からない」を「paired 30-60戦」に圧縮。**L3 の評価コストを 5-10× 削減**。

### 早期シグナル指標 (学習中に監視、ダメなら即中断)
文献の健全性指標を metrics に出し、**閾値を割ったら 2h 待たず kill**:
- **policy entropy**: 1 付近から低下すべき。高止まり=学習失敗。
- **value residual variance**: 1 から急降下すべき。高止まり=価値予測失敗。
- **KL**: 小さい正。大=stale experience。
- **reward scale**: [-10,+10] 内 (逸脱=shaping 係数バグ、iter11 の爆発を即検出)。
- **eval_win trend**: 最初の 5-10 iter で baseline と差が出るか (paired)。

## L4: 本番 GPU (採否確定後のみ)
- L0-L3 を通過し **paired で有意な改善**が出た施策だけ RunPod へ。
- 規約: 中間 best.pt を iter ごと S3 upload、uptime 監視 (memory)。

## 本 project への落とし込み (実装提案)

| # | 施策 | コスト | 効果 |
|:--:|---|---|---|
| 1 | **probe 環境スイート** (L2) を施策ごとに追加 | 低 | 施策ロジックを秒で隔離検証 |
| 2 | **paired-seed A/B 評価ハーネス** (L3) | 中 | 30-60戦で有意差、評価 5-10× 高速化。最重要 |
| 3 | **早期シグナル指標 + auto-kill** | 低 | 無効施策を 2h 待たず中断 |
| 4 | **縮小 config テンプレ** (`fast_probe.yaml`: 20iter/ep8/h500) | 低 | L3 の標準入口 |
| 5 | single-batch overfit テスト | 低 | 新報酬項の実装バグを学習前に弾く |

**核心**: 本 project の「n<300 不信・勝率振動」は **paired-seed 評価**で構造的に解決できる。
これにより「施策の有効性を 20分-2h・30-60戦で有意判定 → 通過分だけ GPU」のパイプラインが成立。
horizon/network/batch を削らず iterations と評価戦数を削るのが、有効性を保ちつつ高速化する鍵。

## Sources
- [Debugging RL (Andy Jones) — probe 環境/single-batch/早期指標](https://andyljones.com/posts/rl-debugging.html)
- [Smoke Testing for ML Pipelines (MLOps Community)](https://home.mlops.community/public/blogs/smoke-testing-for-ml-pipelines)
- [How Many Random Seeds? Statistical Power Analysis in Deep RL](https://arxiv.org/abs/1806.08295)
- [A Hitchhiker's Guide to Statistical Comparisons of RL Algorithms](https://arxiv.org/pdf/1904.06979)
- [Deep RL at the Edge of the Statistical Precipice (NeurIPS 2021)](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)
- [Paired Seed Evaluation (alphaXiv 2512.24145) — 分散低減](https://www.alphaxiv.org/overview/2512.24145)
- [Deep RL Debugging and Diagnostics (Medium)](https://medium.com/swlh/deep-rl-debugging-and-diagnostics-5c9a17e78653)
