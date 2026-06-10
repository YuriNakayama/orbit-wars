# JAX rulebase tournament — JAX 化 case のランキング

`jax_v1/2/3/4/6/8/9`（in-JAX rule agent）を GPU で総当たり対戦させ、勝率 +
Wilson CI でランキングを作る基盤。観測性（逐次ログ・S3・リソース・SSH・事前検証）
を満たし、RunPod で実行する。

## 構成

| ファイル | 役割 |
|---|---|
| `agents_jax.py` | 各 `jax_vN` を vmappable `ActionFn(state)->action` に統一する adapter |
| `selfplay_host.py` | host-driven batch self-play。per-turn graph のみ compile しループは host 側 |
| `run_tournament.py` | round-robin + 逐次 `tournament.json`/`leaderboard.md` flush + S3 + crash-safe + phase timing |
| `configs/{smoke,full}.yaml` | smoke（3 agent/8 seed）/ full（7 agent/300 seed）|

RunPod case: `tournament_rulebase_jax_smoke`（事前検証）/ `tournament_rulebase_jax`（本番）。

```bash
git push origin <branch>
dev/runpod dev <commit> --case tournament_rulebase_jax_smoke   # interactive (SSH 用)
# ready 後 SSH (--via direct) で run_tournament を起動、tail/logs で観測
```

## 実行時間の調査結果（2026-06-09 / RTX 4090 実測 + 解析）

**結論: 現状の agent では「各 300 対戦を 20 分以内」は物理的に不可能。**

### 計測値（RTX 4090, cross-case jax_v1 vs jax_v4）

| batch | per-state | per-turn | 備考 |
|---|--:|--:|---|
| 8 | 3451 ms | 27.6 s | GPU util 89-93%（飽和） |
| 64 | 459 ms | 29.4 s | batch 拡大で per-state **7.5x**（per-turn はほぼ不変）|

- per-turn `compute_actions` は **2304 cell（48×48）× HORIZON scan** の巨大グラフ。
  doc プロファイル: capture 20% / **snipe 40%** / harass 20% / allocator 20%。
- GPU は既に飽和（89-93% util、mem 18/24 GB）。律速は **per-turn の絶対 compute コスト**で、
  host-loop の sync オーバーヘッドではない（G1/G2 で wall-clock 不変を確認済）。

### post-hoist 実測確認（2026-06-10 / RTX 4090）

3 hoist（harass-skip / base_timelines 共有 / snipe-ETA 集約、全て 12/12 一致）適用後に
GPU で再計測したところ、**batch8 × 30 turn が 9 分超**（= **~18.5 s/turn**）。doc の
warm-single 16.3 s/turn とほぼ同じで、hoist による runtime 改善は**僅少**（XLA が既に CSE
していたか compile が支配的）。→ **1 ゲーム（~309 turn）= ~95 分、500 turn = ~154 分**。
300 ゲームは数百時間規模で、**20 分制約に対し ~300x 不足が実測で確定**。

### ゲーム長

both-rulebase self-play の終了 turn: `[498, 175, 481, 282, 125, 295]`（mean 309, max 498）。
batch 実行では早期 break は**最も遅いゲームに律速**されるため大きな短縮にならない。

### なぜ 20 分に収まらないか

総 compute = `games × turns × 2304-cell × cell-cost` は**バッチ化で不変**（GPU 並列の限界に既に到達）。
warm-single = 16.3 s/turn（doc）なので **1 ゲーム（avg 309 turn）≈ 82 分**。300 ゲームでは数百時間規模。
20 分予算に対し ~300x 不足し、batch・hoist では埋まらない。

## 適用済み高速化（全て action 一致性を保持・12/12 検証済）

| # | 施策 | 効果 | commit |
|---|---|--:|---|
| batch | seed を 1 vmap に詰める | per-state **7.5x** | — |
| G1 | host-loop の全 turn sync を排除（async dispatch）| GPU 飽和下では wall-clock 不変 | `01dd39cb` |
| G2 | `donate_argnums`（buffer in-place 再利用）| 同上 | `01dd39cb` |
| hoist1 | `HARASS_ENABLED=False` で harass grid build を skip | case1 ~20% | `e8cba4aa` |
| hoist2 | `_base_timelines` を 1 回計算し grid 間で共有（従来 3 回）| XLA CSE 次第 | `677fbd38` |
| hoist3 | snipe enemy-ETA を per-target に集約（従来 per-cell 48x）| snipe の ETA 部分 | `bf9816aa` |

| vec | `_search_safe_intercept_jax` の 110-cand 逐次 lax.scan を vmap+argmin に置換（web-search 由来）| **悪化 ~18→24.6 s/turn** | `0fc3cb28` |

いずれも `compute_actions(hoist) == compute_actions(original)` を 12 state で bit 一致確認済。

### web-search 由来の最適化（vec）の検証結果（2026-06-10 / RTX 4090）

web search で「独立候補に対する逐次 lax.scan は vmap+argmin に vectorize すべき（GPU 5-20x）」
（jax-ml/jax discussion #10233, VanderPlas）という指針を得て、`_search_safe_intercept_jax` の
110 候補の逐次 scan を `jax.vmap(per_candidate)(cand_grid)` + float32 合成キー argmin に書き換え、
**12/12 bit 一致**を確認。しかし GPU 実測（batch8×30turn）は **24.6 s/turn と逆に悪化**
（vec 前 ~18.5 s/turn）。原因: ① per-turn は 2304-cell の nested vmap が支配的で、その内側 1 scan の
寄与は小さく、② 110 候補を一括 materialize する vmap が compile/dispatch コストを増やし runtime 利得を
相殺。**「逐次 scan を消せば速くなる」は本ワークロードでは成立せず**、律速は inner scan ではなく
grid 全体の compute 量だと実測で確定した。

ただし合計でも per-turn を ~300x 下げるには遠く、**ランキングを 20 分で出すには別アプローチが必要**:

- **agent-level の構造改修**: snipe の `_plan_shot_cell`（aim, per-(s,t) で 40% 級）を per-target 近似に
  集約 = doc#3 の本丸（一致性に影響するため未実施）。または grid の nested-vmap を根本的に削減。
- **スコープ縮小**: 各 pair の対戦数を ~8-16 に下げれば 20 分に近づくが、`n<300` は kaggle 非決定性で
  信頼不可（memory `project_imitation_case1_phase3`）。**20 分制約と n≥300 信頼性は両立しない。**

## 登録済み JAX agent（ランキング対象）

| registry | case | action parity | aim lineage |
|---|---|--:|---|
| jax_v1 | case1 | 90% | 幾何 |
| jax_v2 | case2 | 90% | 幾何 + harass |
| jax_v3 | case3 | 87% | 幾何 + rollout（未 port）|
| jax_v4 | case4 | 100% | エンジン再生 |
| jax_v6 | case6 | 100% | +STAY burst |
| jax_v8 | case8 | 90% | base 戦略 |
| jax_v9 | case9 | 単発 100% | +ANTI_PING_PONG（dormant）|

ランキング結果（勝率表）は本番実行が 20 分制約内に収まる構成が得られ次第ここに追記する。
