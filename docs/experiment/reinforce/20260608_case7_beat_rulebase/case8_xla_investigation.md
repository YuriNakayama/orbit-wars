# case8 agent_jax XLA爆発の調査 (一致度100%・高速化の対象)

時刻: 2026-06-09 / GPU+JAX前提で case8(action 100%一致) を高速化するための原因調査。

## 背景
case8 baseline_jax/agent_jax は本物 v8 と action 100%厳密一致。唯一の課題は compute_actions が
CPU 24,000ms/call で遅く、memory `case_jax_phase0` では GPU bench が XLA compile で 2h ハング。

## サブステージ別プロファイル (CPU, cached)
| ステージ | 時間 | 割合 |
|---|---|---|
| build_features_from_state | ~6ms | ~0% |
| build_capture_grid | 4,877ms | 20% |
| build_snipe_grid | 9,789ms | 40% |
| build_harass_grid | 4,889ms | 20% |
| run_mission_and_followup (allocator) | ~5,000ms | 20% |

→ ボトルネックは allocator でなく grid builders (~80%)。

## 根本原因: 2304セル × HORIZON scan の nested vmap
- grid builders は per-(src,tgt) を nested vmap = 48×48 = 2304 セル。
- 各セルが plan_shot/aim を呼び、その中に HORIZON(=110)turn の lax.scan が複数 + REFINE_ITERS=5。
- allocator も _need_with_commits が mission scan(2304候補)内で HORIZON timeline scan を回す (scan-in-scan)。
- = 2304セル × ~110step scan × 3 grid の巨大融合グラフ → CPU 24s / GPU compile 2h ハング。

## GPU での見込みと最適化方針
- CPU 24s は逐次 vmap の値。GPU では 2304 セルが並列実行されるので runtime は大幅短縮見込み。
- 真の blocker は XLA compile の巨大グラフ (2h ハング)。

### 最適化候補 (一致度100%維持)
1. 冗長計算の hoist: _base_timelines(per-planet 48個)を共有し per-cell scan を削減。
2. HORIZON scan の短縮: cutoff(到達turn~20-30)までで十分だが固定長110で回している。
3. grid の per-cell aim を per-target に集約 (src×tgt のうち target 共有)。
4. compile 分割: 3 grid を別 jit にし融合グラフを分割。

## 次のステップ
- GPU pod で case8 compute_actions の 実 compile時間 + runtime を実測。
- compile が現実的なら runtime短縮を、2hハング再現なら grid の nested vmap 削減を優先。

## GPU 実測結果 (2026-06-09, RTX 4090 SECURE)

bench: `pipeline/rulebase/case8/baseline_jax/_gpu_bench.py` (commit 7516d2f2)。
RunPod RTX 4090 24GB に CUDA jaxlib (`jax[cuda12]==0.10.0`) を venv 導入し実測。

| 測定 | 値 | 意味 |
|---|---|---|
| platform | gpu (cuda:0) | GPU 認識 OK |
| **compile_s** | **28.4s** | ★2h ハングは再現せず。GPU では ~30s でコンパイル完了 |
| warm_single_ms | **16,291ms** | 1 call ~16.3s。CPU 24s からほぼ改善せず (per-cell scan は逐次) |
| vmap8 compile | 32.8s | batch 化しても compile は ~30s 据え置き |
| vmap8 per_state_ms | 2,092ms | 8 並列は 4090 を使い切れず |
| vmap64 compile | 34.8s | |
| **vmap64 per_state_ms** | **283ms** | ★batch64 で 1 state あたり 283ms = single の 57倍速 |

### 結論 (2 つの問いへの回答)
1. **compile は 2h ハングしない** — GPU で 28-35s。memory `case_jax_phase0` の「2h ハング」は
   CPU compile の値であり、GPU では現実的。XLA 巨大グラフ説は「compile が長い」点は正しいが
   「実用不能」は誤り。
2. **GPU vmap は効く (ただし batch 必須)** — single call は 16.3s で CPU 24s からほぼ改善なし
   (nested scan が逐次なため)。**batch64 vmap で 283ms/state まで短縮 (57倍)**。
   → case8 を学習/評価の opponent に使うなら **必ず vmap で複数 env を束ねる**こと。
   self-play harness は元々 vmapped なので構造的に整合する。

### 含意
- **最適化(grid 分割 / HORIZON 短縮 / base timeline hoist)は必須ではない**。batch vmap で
  283ms/state は実用域 (1 turn 1s 予算に対し margin あり、学習 rollout なら GPU 並列で更に償却)。
- 単発 actTimeout=1s の Kaggle submit 用途では single 16.3s が致命的 → **submit には不向き**、
  あくまで **GPU 上の学習/評価 opponent (batch 前提)** としての用途に限る。
- 次に optimize するなら per_state を更に下げる余地: vmap128/256 で 283ms から漸減するか、
  HORIZON cutoff で compile グラフ縮小 (compile 30s→数s) が候補。だが現状でも opponent 用途は成立。
