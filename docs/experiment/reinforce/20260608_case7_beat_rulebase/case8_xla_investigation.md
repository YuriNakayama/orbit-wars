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
