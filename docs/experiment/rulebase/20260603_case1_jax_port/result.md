# rulebase/case1 JAX full port — result (経過 1)

> 記録: 2026-06-03 01:45 / 状態: in_progress (ループ cron c0a10ea6, 10m)

## この時間の進捗 (Step1 確立 → core_jax 実装)

### Step1: 結合テスト確立 ✅
- `tests/e2e/pipeline/rulebase/case1/test_agent_jax_identity.py` (JAX vs 元 Python)。
- 1試合 ≈ 18.6s (要件「10分以内」クリア)。
- 3 テスト: action 等価 (5 seed, 完全一致) / smoke / **0勝 tripwire** (4game, 最低1勝)。
- 現状 lite port: action 一致率 **20.9%**, tripwire **0/4 勝** = 回避すべき失敗モードを再現中。

### Step2 方針 + core_jax 実装 (ボトムアップ, x64 parity)
| モジュール | parity | 内容 |
|-----------|--------|------|
| geometry_jax | ✅ 13/13 | dist/segment/sun-safe/safe_angle |
| physics_jax | ✅ 20/20 | fleet_speed(log)/is_static/predict_position/estimate_arrival |
| aim_jax | ✅ 8/8 | aim_with_prediction (5-iter lax.scan refine + search fallback), static+rotating |
| worldmodel_jax | ✅ 10/10 | resolve_arrival_event + keep_needed (110-turn fold + 並列候補) |
| **合計** | **✅ 51/51 GREEN** | core(数値+防衛)層は本物と x64 完全一致 |

全 lint + mypy clean。

## 重要な発見

1. **lite port の fleet_speed が本物と別物** (`max(0.5,2-0.05√)` vs case1 `1+(6-1)*ratio^1.5`)。
   速度が違えば到達ターン→必要数→action が全ズレ = **0勝の確実な一因**。正しい formula で parity 達成。
2. PoC2 で「固定長 scan で表現可」とした防衛 timeline が、keep_needed の 110-turn `lax.scan`
   fold + 全 keep 候補並列評価として実装でき、本物の binary search と完全一致。
3. web research の知見 (scan early-exit + masked argmin) が aim_jax の refine 実装で有効だった。

## NEXT ACTION

- **missions_jax**: option_collector の score chain (target_value/preferred_send/
  opening_filter/apply_score_modifiers) を JAX 写経。~40 定数。PoC1 で per-target
  score+argmax 構造を実証済。
- その後 **mission_resolver_jax** (固定長 lax.scan、PoC2 で実証済の逐次 greedy)。
- 統合後、結合テストの action 一致率 (21%→) と tripwire (0/4→) が動き出す = 劣化解消の実測。

## 切り戻し点
- commit `014bcb9` (計画のみ) / `cfbfb86d` (core_jax 完成) が安全点。

---

# 経過 2 (2026-06-03 ~02:00)

## core_jax 全部品 parity 完成 + 初の agent 統合

- featurize_jax (reaction_times 7/7) + missions_jax (score chain 10/10) 追加 → **core_jax 全 68 parity GREEN**。
  parity test が定数 2 件の写し間違いを検出 (TDD 奏功)。
- **agent_full_jax** (capture-single-source slice) を core_jax 部品から統合:
  - **turn0: 20/20 match** (本物と launch/hold+target+send 完全一致) ← 初の full pipeline 一致
  - **full-game: 3.8%** (19/498, seed0) ← capture-only のため中盤で乖離

## 重要な所見 (正直な評価)

turn0 が 20/20 でも full-game は 3.8%。これは capture-single-source slice が
**reinforce / swarm / crash / followup / evac + commitments 累積 + keep_needed
reserve** を未実装のため。lite port (full-game 20.9%) を一時下回るのは、lite の
reserve ヒューリスティックが中盤を粗く拾っていた分。**full-game 一致には全 mission
種 + mission_resolver 固定長 scan の統合が必須**と実測で確認。

## NEXT ACTION

1. **available に keep_needed reserve を wire** (arrivals を EnvState fleets から構築)
2. **commitments 累積** を mission_resolver 固定長 scan で実装 (PoC2 構造)
3. reinforce / swarm / crash / followup / evac mission を順次追加
4. 各追加ごとに full-game 一致率の上昇を結合テストで実測 (3.8% → 100% を目指す)
5. dtype: jnp.float_ の x64 警告は無害 (float32 truncation) だが production では float32 明示推奨
