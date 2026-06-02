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

---

# 経過 3 (2026-06-03 ~02:15) — 戦略の見直し

## 2つの気づき (実装を進めて判明)

### (A) 目標の再定義: 100% byte-parity ではなく「劣化しない (≥1勝/4)」
ユーザーの核心要求は「JAX化で勝率ほぼ0になるのを避ける」。100% action 一致は
**手段であって目的ではない**。0勝 tripwire (4game, ≥1勝) が真の受け入れゲート。
→ capture-slice agent で tripwire を実測し、既に勝てるなら full byte-parity は
過剰投資の可能性。全 mission 写経の前に「勝てるか」を先に測る。

### (B) 速度: jit 未適用が問題
agent_full_jax を per-call で呼ぶと遅い (tripwire 4game が 2分でも未完)。
ユーザー要件「1試合10分以内」と GPU vmap 目標の両方に **jax.jit ラップが必須**。
core_jax は全て jit/vmap-friendly に書いてあるので compute_actions_jax を
`jax.jit` でラップするだけ。これは速度と劣化検証の両方に効く。

## NEXT ACTION (改訂)

1. compute_actions_jax を `jax.jit` ラップ → 1試合速度を計測 (要件 10分以内)
2. tripwire (4game) を jit 版で実測 → **既に≥1勝なら劣化問題は解決済**かを判定
3. 勝てない/不足なら mission 種を追加 (keep_needed reserve → commitments → reinforce…)
   し、各追加で tripwire と full-game 一致率の両方を実測
4. 「劣化なし (tripwire GREEN) + 高速 (jit, 10分以内)」が達成基準

## 実測判定 (経過3 の問い (A) への回答)

**capture-slice agent: tripwire 0/4 勝** (seed0/1 × 2 seat 全敗、勝者は常に Python 側)。
→ **byte-parity を諦めて capture だけで勝てる、という近道は否定された**。capture-only は
本物に対し依然「勝率ほぼ0」= 回避すべき失敗モードそのまま。**full mission set の
忠実 port が劣化回避に必須**と実測で確定。turn0 一致 (20/20) は必要条件だが十分でない。

→ 方針確定: 近道なし。keep_needed reserve → commitments → reinforce → swarm →
crash → followup → evac を順次 port し、各段で tripwire を実測。tripwire が ≥1勝に
転じた時点が「劣化解消」の最小到達点、full-game 100% 一致が完全到達点。

---

# 経過 4 (2026-06-03 ~02:35) — 速度クリア + reserve wire

## 達成
- **jit-wrap**: compute_actions_jax_jit (seat static)。compile 0.61s / warm 3.77ms/call /
  1 full game 14.3s。**「1試合10分以内」クリア**。turn0 parity jit でも 20/20。
- **fleet_target_planet** (ray-circle, 5/5 parity, 750 fleet) + **build_arrival_ledger** +
  **compute_reserve_per_planet** (keep_needed reserve, no-arrival 短絡)。
- agent に reserve wire: available = ships - keep_needed reserve。turn0 20/20 維持 (回帰なし)。

## 実測 (正直)
| 指標 | capture-only | +reserve |
|------|-------------|----------|
| turn0 一致 | 20/20 | 20/20 |
| full-game 一致 | 3.8% | **4.0%** (僅差) |

reserve 単体では full-game 一致は動かない。**支配的な乖離は mission 種 (reinforce/swarm) +
commitments 累積** (sorted-mission loop で need/target が変わる) と判明。reserve は防衛で
必要だが勝率の主因ではない。

## NEXT ACTION
1. **mission_resolver 固定長 scan + commitments 累積** (PoC2 構造) を agent に組込む —
   これが target 選択順と need を本物に揃える本丸。
2. **reinforce mission** 追加 (自陣防衛の launch、勝率に直結)。
3. 各段で tripwire (4game) と full-game 一致を実測。tripwire ≥1勝 = 劣化解消の最小到達。
4. parity 73/73 GREEN 維持。core_jax 部品は揃ったので残りは「組み上げ」。

---

# 経過 5 (2026-06-03 ~02:55) — 乖離原因を特定 (over-fire bug)

## resolver scan 実装 → full-game 4.0% 変わらず。診断で原因特定

mismatch を分類 (seed0, 本物プレイ 498 turn):
| 種別 | 件数 |
|------|------|
| agree | 20 |
| **py_hold / jax_fire (JAX が余計に撃つ)** | **230** |
| py_fire / jax_hold | 0 |
| both_fire_diff | 248 |

→ **JAX agent は過剰発射 (over-fire)**。Python が hold する 230 turn で JAX は launch。
JAX が誤って hold することは皆無 (0)。これが低一致率 + 0勝 (ship 過剰投入で自陣手薄) の根本原因。

## 原因の仮説 (高確度)

JAX の `need = ceil(target.ships)+1` は **in-flight 友軍 fleet を無視**。本物は
`ships_needed_to_capture` → `projected_state` → `base_timeline[target]` (=
`arrivals_by_planet` 由来、在空 fleet 込み) で投影するため、既に友軍 fleet が捕獲に
向かっている target は need=0 → hold。JAX はこれを見ず再発射 → over-fire。

## NEXT ACTION (precise fix)

1. **need に in-flight 投影を wire**: build_arrival_ledger (実装済) で各 target の
   到達 fleet を集計し、keep_needed/timeline と同じ simulate で projected owner/ships を
   出す。owner==me なら need=0。これが over-fire の直接修正。
2. 修正後に mismatch 分類を再測 (py_hold/jax_fire が減るか)。
3. tripwire も再測 (over-fire 解消で 0勝脱出を期待)。
4. これが効けば「劣化なし」に最短到達。reinforce 等はその後。
