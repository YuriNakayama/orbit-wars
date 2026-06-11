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
| `compare.py` | **二値比較 primitive**「A>B か?」。両座席 100 戦ずつ→Wilson 95% CI で `a_wins`/`b_wins`/`tie` 判定 |
| `ranking.py` | **最小コスト ranking driver**。事前順位の隣接ペアを検証 + 同点マージ + 反転時 insertion-sort 修復。逐次 `ranking.json`/`ranking.md` flush + S3 + crash-safe |
| `configs/{smoke,full}.yaml` | round-robin: smoke（3 agent/8 seed）/ full（7 agent/300 seed）|
| `configs/ranking{,_smoke}.yaml` | ranking: full（prior 6 agent/100 seed/h300）/ smoke（3 agent/8 seed/h120）|

RunPod case: round-robin は `tournament_rulebase_jax{,_smoke}`、ranking は `ranking_rulebase_jax{,_smoke}`。

```bash
git push origin <branch>
dev/runpod dev <commit> --case ranking_rulebase_jax_smoke   # interactive (SSH 用)
# ready 後 SSH (--via direct) で ranking を起動、tail/logs で観測 → 本番は ranking_rulebase_jax
```

## ranking 方式（最小コスト・2026-06-10）

round-robin（21 ペア×300 戦＝数百時間）は時間制約下で不可能（下記実測）。代わりに
ユーザー確定の 3 前提を使い、**比較ソート**に置き換える:

1. **相性なし=全順序（推移的）** → ranking は SORTING。検証済み順序は n-1 隣接比較で足りる。
2. **番号大=強（強い事前順位）** → 順序を*発見*せず**検証**。prior `[v8,v6,v4,v3,v2,v1]` の
   隣接 5 ペアを確認、全一致なら推移律で全順位確定。
3. **ほぼ同強が存在** → 分離に games を浪費せず**同点バケットにマージ**。

`ranking.py` は prior に対する insertion-sort: 各 agent を直前の確定 bucket と比較し、
`A_WINS`→下位に新 bucket / `TIE`→同 bucket マージ / `B_WINS`（反転）→上位へ bubble して
局所修復。比較は `compare_pair` の固定 200 戦二値判定（SPRT 早期停止なし）。

- 除外: `jax_v9`（dormant ≈v8）, `jax_v7`（strict port 未完: STAY/ACCUMULATE pending ≈v8）。
- horizon=300（500 でなく）: ranking は勝敗の**符号**のみ必要。較正（h300 vs h500 で勝者一致）で検証。
- ロジックは `tests/unit/pipeline/rulebase/_bench/test_ranking.py` で検証済（prior 一致 / 同点マージ /
  反転修復 / bubble-to-top / 全同点 / 暴走 abort の 7 ケース）。

### GPU 実測結果（2026-06-10 / RTX 4090 / smoke `ranking_rulebase_jax_smoke`）

ranking driver を pod で dry-run（3 agent `[v8,v4,v1]` / 8 seed/half / **h120** / 16 戦/比較）した結果:

- ✅ **観測性は機能**: 起動直後に `ranking.json`/`ranking.md` を flush（crash-safe 逐次書き込み確認）。
  SSH `--via direct` + `--exec` でのリアルタイム観測も動作。
- ❌ **しかし第1比較（v8 vs v4, 16 戦, h120）が ~23 分経っても未完了**（GPU 98% で compute-bound、
  hang ではない）。ranking は比較*回数*を 21→5 に削減するが、各比較は依然 per-turn ~18.5s の
  2304-cell grid コストを払う。h120 の最小比較ですら ~23 分超 → **本番 h300（turn 2.5x）× 100 seed ×
  6 比較は数十時間規模**で、当初見積 ~20h すら楽観的だった。

**結論**: 比較ソート化（アルゴリズム改善）は正しく機能するが、**律速は per-turn の grid compute
であり比較回数ではない**ため、ranking 法でも「各 300 対戦を 20 分以内」は達成不可能。loop 指令
「長時間化したら中断」に従い smoke を中断・pod 停止（課金 ~$0.3）。20 分制約を満たすには ranking 法
ではなく、agent 側の **grid nested-vmap 自体の削減**（action 一致性に影響、別タスク）が必須。
→ 既存の [[project_jax_strict_tournament_perturn_bound]] の結論を再確認。

## ★ allocator truncation の GPU 実測（2026-06-11 / RTX 4090）— 幾何系で 15-18x 達成

「flops 不変=デバイス非依存で無効」とした先行判断は**誤り**だった。GPU の per-turn コストは
flops ではなく**逐次カーネル起動チェーン**（scan step 数 × 起動レイテンシ）であり、allocator の
N=4608 scan を top-K に truncate（恒等保存: tail は `-inf` no-op、回帰テスト 2/2 + 40/40 state で
action byte 一致）すると **GPU で激減**する:

| K | batch | per-turn | speedup |
|---|--:|--:|--:|
| 4608（原本）| 8 | 11.12 s | 1.0x |
| 256 | 8 | 1.06 s | 10.5x |
| **64** | 8 | **0.61 s** | **18.3x** |
| 256 | 64 | 1.20 s | 9.3x |
| **64** | 64 | **0.72 s** | **15.5x** |

`MAX_ALLOC_CANDIDATES=64`（valid 候補実測 max=10 の 6x マージン）を**全8 case** の allocator に
適用済（commit `fe658c65` 系列）。CPU では 1 step ≈ μs（起動 overhead 無）のため効果が見えず
1.00x — **CPU 計測で GPU 効果を否定してはならない**（本件の最重要教訓）。

### ランキング本番（h500 / 100seed×2席 / 隣接5比較）→ エンジン再生系で中断

- 幾何系 (case1) は 0.61-0.72 s/turn を確認しゲート（≤3s）通過。
- しかし本番 cmp1 = **jax_v8 vs jax_v6（エンジン再生 aim 系×2）が ~3.7h でも未完**（GPU 85-89% で
  健全計算中、per-turn **>13 s**）。truncation 後もエンジン再生系には allocator 以外の長い逐次
  チェーン（engine-sweep aim / harass grid / STAY 等）が残存し、**aim 系統間で per-turn が ~18x 乖離**:

| aim 系統 | per-turn (K=64) |
|---|--:|
| 幾何（case1/2/3）| 0.61-0.72 s |
| エンジン再生（case4/6/8/9）| **>13 s** |

- projection >14h のため loop 指令に従い**中断・pod 停止**（課金 ~4h ≈ $2.8）。
- jaxpr 解析で乖離を定量化: **case8=6,831 逐次 scan step vs case1=1,978**（case8 は 110×44 本 +
  **219×8 本の half-step sweep** が追加）。逐次 3.5x × per-step の重さ ≈ 観測 18x と整合。
- **次の一手**: エンジン再生系の逐次チェーン削減（重複 110-scan の共有化など、識別性ゲート付き）
  → 全 6 agent ランキング再実行。

## ★ 幾何系部分ランキング結果（2026-06-11 / RTX 4090 / h500 / 100 seed×2席）

truncation 済み幾何系 3 agent のランキングが **28 分で完走**（2 比較 × 200 戦 = 400 戦）:

| rank | agents |
|---|---|
| 1 | **jax_v3, jax_v2, jax_v1（統計的同強・1 バケット）** |

| # | A | B | A勝 | B勝 | win% | 95% CI | verdict |
|---|---|---|--:|--:|--:|---|---|
| 1 | jax_v3 | jax_v2 | 100 | 100 | 0.500 | [0.431, 0.569] | tie |
| 2 | jax_v3 | jax_v1 | 99 | 101 | 0.495 | [0.426, 0.564] | tie |

- v3-v2 が tie → 同バケット → 代表 v3 と v1 を比較 → tie → **3 agent 全てが 1 バケットに収束**。
  「ほぼ同強の case が存在する」という事前知識と整合（case1/2/3 は同系統の漸進改良）。
- 実測スループット: per-half 414-424s（100 戦, h500）= **0.83s/turn @batch100**。
  **1 比較 ≈ 14 分 → 「各 300 対戦を 20 分以内」は幾何系で達成**（300 戦でも batch を埋めれば
  turn 数は不変のため wall-clock ほぼ同じ）。
- 総コスト: pod ~45 分 ≈ $0.5。

## ★★ 最終ランキング結果（2026-06-11 / 全 6 JAX agent / h500 / 各 200 戦）

エンジン再生系の **219×110 ネスト scan**（`_search_safe_intercept` 内の engine-hit scan、
~24,000 逐次起動 = エンジン系 18x 遅の真因）を vmap+argmin に vectorize（恒等 32/32、
OOM は `lax.map(batch_size=32)` で解消）した結果、エンジン系比較も ~20-25 分/比較で完走:

| # | 比較 | 戦績 | win% CI | verdict | 所要 |
|---|---|---|---|---|--:|
| 1 | v3 vs v2 | 100-100 | [0.431, 0.569] | tie | 14 分 |
| 2 | v3 vs v1 | 99-101 | [0.426, 0.564] | tie | 14 分 |
| 3 | v8 vs v6 | 97-103 | [0.417, 0.554] | tie | 24.5 分 |
| 4 | v8 vs v4 | 100-100 | [0.431, 0.569] | tie | 24.6 分 |
| 5 | v8 vs v3 | 101-99 | [0.436, 0.574] | tie | 19.6 分 |

### 最終順位表

| rank | agents |
|---|---|
| 1 | **jax_v8, jax_v6, jax_v4, jax_v3, jax_v2, jax_v1（全 6 agent 統計的同強・単一バケット）** |

- 隣接 4 比較 + 連結比較（v8-v3）の全てが tie。**「case 番号大 ≈ 強」の事前順位は n=200/比較の
  解像度（CI 幅 ±7pp）では検出されない** — 過去の ablation が示した case 間差（±数 pp）は
  この検出閾値未満であり、「ほぼ同強の case が存在する」という事前知識が全ペアに該当した。
- より細かい順位付けには 1 比較あたり n を桁で増やす必要がある（±2pp 分解能 ≈ n=2,400/比較。
  現速度なら 1 比較 ~4-5 時間で実行可能）。
- 総実行時間: 幾何 2 比較 28 分 + エンジン 3 比較 69 分 ≈ **97 分 / 1,000 戦**（コスト ~$2）。
  当初の「各 300 対戦を 20 分以内」は幾何系ペアで達成、エンジン系ペアは ~25 分/200 戦。

### 適用した高速化の最終台帳（全て action 恒等・ゲート付き）

| 施策 | 恒等性 | GPU 効果 |
|---|---|---|
| allocator scan top-K=64（全 8 case）| 2/2 pytest + 40/40 | 幾何系 **15-18x**（11.1s → 0.61-0.83s/turn）|
| aim first-hit scan → vmap+argmax（4 case）| 32/32 | ネスト解消に寄与 |
| 219 候補 intercept → lax.map+argmin（4 case）| 32/32 | エンジン系 **~9x**（>13s → ~1.5s/turn）|
| grid live-source gather（case1）| 96/96 | 1.05x（小）|

### 残課題: エンジン再生系の逐次チェーン削減（帰属解析済・実装待ち）

`engine_scan_attribution.py` による case8 のコンポーネント別逐次 step 帰属:

| component | 逐次 steps | 内訳 |
|---|--:|---|
| build_{capture,snipe,harass}_grid | 1,684×3 | 各 **219×2 + 110×11**（cell 内 plan_shot×2 起因）|
| run_followup_pass | 1,602 | followup も plan_shot を実行（219×2 + 110×10）|
| 1 plan_shot | 667 | **219×1（half-step engine sweep）+ 110×4（aim + 安全ゲート3種が別 scan）** |
| _run_mission_scan | 174 | ✅ K=64 truncate 済 |

識別性保存の削減設計（推定合計 ~70% 減 → per-turn ~13s → ~3s 圏）:
1. **ゲート fused-scan 統合**（−2,640 steps）: aim / sun-safe / intercept / comet-cross の 110-scan
   4 本は同じ turn 軸を独立に走るため、tuple carry の 1 scan に統合しても同値。
2. **219 half-step sweep の共有**（−~1,500）: sweep が ships 非依存（target 軌道のみ依存）なら
   per-target に hoist して 3 grid × 2 call で共有可（要依存性確認）。
3. **base_timelines hoist**（−700）: case1 で実証済みの doc#1 hoist を case8 系へ。

実装順: case8 で 1→3→2 を適用 → 恒等テスト（hybrid vs full の action 比較）→ GPU smoke で
per-turn 確認 → case4/6/9 へ横展開 → 残り 3 隣接比較（v8-v6, v6-v4, v4-v3）で全 6 agent 完成。

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
