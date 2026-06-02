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

---

# 経過 6 (2026-06-03 ~03:15) — 残 over-fire の精密診断

## projection fix で 4%→11%。残 over-fire 195 を深掘り

具体例 (seed0 t32, src=24 が 6 ships 発射、Python は hold):
- src=24 の到達可能 target を JAX need で評価:
  | target | ships | #in-flight | proj@5turn | need@5 |
  |--------|-------|-----------|-----------|--------|
  | 19 | 9 | 0 | neutral,9 | 10 |
  | 22 | 20 | 0 | neutral,20 | 21 |
  | 6 | 45 | 0 | neutral,45 | 46 |
  | **26** | 14 | **1 (enemy)** | **enemy,2** | **3** |
- → JAX は target26 を「敵 fleet が削るので need=3、6 ships で安く取れる」と判断し発射。
  **Python は hold** (敵が捕る寸前の planet を横取りしない / score が低い)。

## 所見

残 over-fire は **gross bug でなく nuanced な score/guard 差**。JAX は enemy-contested
planet を「安い snipe」と見て撃つが、本物は撃たない。原因候補:
(a) target_value に indirect_wealth=0 で渡している (本物は非0) → score 過大
(b) snipe mission の score modifier / build_snipe_mission の閾値未 port
(c) 敵が捕る planet への横取り抑制ロジック未 port

projection fix 自体は正しい (11% へ前進)。次は (a) indirect_wealth を wire (score 精度)
→ 効果を full-game 一致 + py_hold_jax_fire で測る。

## NEXT ACTION
1. indirect_wealth_map を JAX 化し target_value に wire (score 過大の最有力原因)
2. 再診断 (py_hold_jax_fire が減るか)
3. 残れば snipe / 横取り抑制を調査 (web search も可)

---

# 経過 7 (2026-06-03 ~03:35) — indirect_wealth wire (仮説a 棄却)

## indirect_wealth を JAX 化・wire。parity 0 mismatch だが over-fire 変わらず

- `featurize_jax.indirect_wealth` 実装、実盤面 5 seed で WorldModel.indirect_wealth_map と
  **0 mismatch**。target_value に wire (これまで 0 を渡していた)。
- 実測: turn0 20/20 維持、full-match **11.0% で不変**、py_hold_jax_fire **195 で不変**。
- → **仮説 (a) score 過大の原因は indirect_wealth ではない、と棄却**。indirect_wealth 自体は
  正しく必要 (full-game score 精度に寄与) なので採用・保持。

## 次の調査方針 (over-fire target26 の Python veto 源を直接特定)

t32 target26 は enemy-contested neutral (my_t=42 >> enemy_t=9, need=3 by JAX)。
Python が hold する真因を、**本物 option_collector を src=24→target26 で直接呼んで
どの guard で reject されるか**を切り分ける (score 比較でなく veto 箇所の特定)。
候補: build_snipe_mission 経路 / target_value が ≤0 / send_cap<needed の partial guard /
是非 enemy が捕る planet への抑制。次イテレーションで実施。

---

# 経過 8 (2026-06-03 ~03:55) — over-fire 真因 = commitments 過剰減算

## 本物 option_collector 直接トレース (src24 t32)

- **本物**: src24→tgt26 で turns=42, needed=40, send_cap=6 → **6<40 で不採用 → hold**。
  JAX aim も turns=42 で一致 (aim は正しい)。
- **JAX が実際に撃つ先 = tgt6** (45 ships, prod4, turns=47)。JAX need(tgt6@47)=**46** (正しい)。
  send_cap=6。本来 6<46 で ineligible のはず。
- なのに発射 → **resolver の `need_now = max(0, f_need - committed[tgt])` が committed で
  need を過剰に削っている**。高 score の別 option が tgt6 に committed を積み、src24 の
  need_now が 6 以下に下がって発射。

## 真因 = multi-source allocation 未 port

本物の `process_multi_source_mission` (mission_resolver.py:84) は、複数 source が
**1つの mission の options として束ねられた時だけ** 協調 send する。私の resolver は
**全 (src,tgt) を独立 option 化し committed を無条件累積**したため、本来 multi-source
mission に束ねられない単独 capture が「他者の committed」で安く見えて誤発射。

→ 修正方針: committed を「同一 mission 内の協調」に限定する。最小修正は **commitments を
撤廃し、単一 source が単独で need を満たす場合のみ発射** (process_single_source_mission
の semantics: `send_limit < missing → return`)。multi-source swarm は後で別途。

## NEXT ACTION
1. resolver の committed 累積を撤廃 → fire 条件を `send_cap >= f_need` (単独充足) に。
2. 再診断 (over-fire 195 が減るか) + turn0 20/20 維持確認。
3. 効けば full-game 一致が上がり tripwire 改善を期待。multi-source は swarm port で対応。

---

# 経過 9 (2026-06-03 ~04:15) — over-fire 真因 確定: plan_shot guards 未 port

## committed 撤廃でも不変 → 真因は aim でなく plan_shot の追加 guard

- committed 過剰減算説 → 修正しても 11.0%/195 不変で棄却。
- 決定打: **`plan_shot(24,23,6) = None` (Python)** だが JAX aim_with_prediction は
  `aim_ok=True`。同 trajectory で可否が逆。
- 原因確定: 本物 `WorldModel.plan_shot` (world_model.py:537) は aim_with_prediction の
  後に **4 つの guard** を適用、JAX agent はこれを skip:
  1. `is_trajectory_sun_safe(launch_x, launch_y, angle, turns, ships)` — turns 全体の sun 横断
  2. `intercept_holds_within_tolerance` — 移動 target が tolerance 内に留まるか
  3. `target_reachable_before_comet_expiry`
  4. `fleet_crosses_other_comet`
- JAX は aim 内の per-segment sun check のみ → **full-trajectory sun + intercept tolerance を
  見ず、撃てない弾道を撃てると誤判定 = over-fire の主因**。

## NEXT ACTION (確度高)
1. **is_trajectory_sun_safe + intercept_holds_within_tolerance を JAX 化**し plan_shot
   相当の wrapper を aim_with_prediction の後段に追加 (comet 2 guard は非 comet では自明 True)。
2. x64 parity test (本物 plan_shot vs JAX wrapper)。
3. agent に wire → over-fire 再測 (195 が大幅減を期待) + turn0 20/20 維持。
4. これが本命修正。effけば full-game 一致と tripwire が同時改善する見込み。

---

# 経過 10 (2026-06-03 ~04:35) — plan_shot guards 修正で大躍進 🎯

## safety_jax (is_trajectory_sun_safe + intercept_holds_within_tolerance) wire

- `safety_jax.plan_shot_ok` を実装し agent の aim_ok に AND。
- **実測 (劇的改善)**:
  | 指標 | before | **after** |
  |------|--------|-----------|
  | full-game 一致 | 11.0% | **49.6%** (4.5×) |
  | py_hold_jax_fire (over-fire) | 195 | **13** (-93%) |
  | turn0 一致 | 20/20 | 20/20 |
- **over-fire の主因は plan_shot guards 未 port で確定**。診断駆動 (経過5-9) が的中。

## 新たな乖離: pf_jh=154 (JAX が hold しすぎ)

over-fire 解消の裏で **py_fire/jax_hold が 0→154 に増加**。JAX が本物より保守的に
なった = guards (or aim_ok) が本物が撃つ shot も一部 reject。次の調査対象。
both_diff=84 も残る (撃つが内容違い)。

## 🎯 tripwire GREEN: JAX won 2/4 vs Python — 0勝問題 解決

**tripwire 0/4 → 2/4** (mirror-ish match で期待通りの ~50%)。**ユーザーの核心要求
「JAX 化で勝率ほぼ0」が解決**。over-fire (ship 過剰投入で自陣手薄) が 0勝の主因で、
plan_shot guards 修正でそれが解消 → 勝てるようになった。

達成基準の到達状況:
| 基準 | 状態 |
|------|------|
| 高速 (1試合≤10分) | ✅ jit 1game 14.3s |
| 劣化なし (tripwire ≥1勝) | ✅ **2/4** |
| full-game action 100%一致 | 進行中 (49.6%) — 完全一致は残課題だが劣化問題は解決済 |

## NEXT ACTION (劣化は解決、精度向上フェーズ)
1. pf_jh=154 の診断: JAX が hold する turn で Python が撃つ target を特定
   (reinforce mission 未実装で hold? guards 厳しすぎ?)
2. full-game 一致 49.6% → さらに上げ、tripwire を 300戦規模で確認 (劣化が確実にないか)
3. reinforce / swarm mission 追加で完全一致に近づける。

---

# 経過 11 (2026-06-03 ~04:55) — pf_jh=154 の原因 = reinforce 未実装

## pf_jh (JAX hold / Python fire) turn の Python move 分類

pf_jh 154 turn の Python move を target owner で分類:
- **reinforce (own-target, 自陣へ送る) = 300**
- capture (enemy/neutral) = 49

→ **JAX が hold しすぎる主因 = reinforce mission 未実装**。本物は threatened な自陣 planet へ
ship を送る (防衛) が、JAX は capture only なので撃たず hold。capture 系の 49 は guards が
やや厳しい分 (副次)。

## reinforce port のスコープ

`build_reinforcement_missions` (108行) は `world.threatened_candidates`
(= _compute_defense_buffers が holds_full=False の自陣 planet を fall_turn/deficit_hint
付きで抽出) に依存。これは keep_needed timeline の派生情報。port には:
1. threatened_candidates 相当 (各自陣 planet の fall_turn + deficit) を JAX timeline から抽出
2. src→threatened target の reinforce option (source_inventory_left ベース、send=missing+margin)
3. resolver に reinforce mission 種を追加 (single-source path)

## tripwire 拡大確認: JAX won 8/10 vs Python (5 seed × 2 seat)

bounded sample (10 game) で **8/10 勝**。0勝問題は決定的に解決、むしろ現状 JAX が
やや勝ち越し (capture 寄りで aggressive、reinforce skip でも短期戦で有利)。

**劣化なし は完全達成**。ただし「baseline_v1 の忠実 opponent」目的では byte-parity
(49.6%→100%) がまだ要る。anti-degradation goal は ✅、parity goal は進行中。

## NEXT ACTION
1. reinforce mission を port (threatened 抽出 + reinforce option + resolver 統合) →
   pf_jh 減・full-game 一致向上
2. その後 swarm/crash/followup/evac で byte-parity を 100% に寄せる
3. parity 100% 後、GPU で vmapped self-play 速度 bench (RunPod)

---

# 経過 12 (2026-06-03 ~05:15) — reinforce wire (inert, 要 avail 修正)

- threatened_info (parity 0) で is_threatened 抽出 → pair() に reinforce 分岐追加。
- 実測: turn0 20/20 + tripwire 2/4 維持 (回帰なし)、だが full-match 49.6%/pf_jh 154 不変
  = reinforce 未発火。
- 原因: resolver avail_now = available(ships-reserve) を使うが reinforce は
  source_inventory_left (ships-spent, reserve非減算) を使うべき。threatened source は
  reserve で available≈0 → 送れない。
- NEXT: resolver に mission_kind を持たせ reinforce 予算を ships-spent に分離 → 再測。

---

# 経過 13 (2026-06-03 ~05:35) — reinforce 予算分離 + pf_jh 主因の訂正

## reinforce budget を kind 別に (capture=available, reinforce=ships-spent)

- resolver に f_reinf flag を追加、reinforce は source_inventory_left (ships-spent,
  reserve 非減算) を予算に。turn0 20/20 + tripwire 2/4 維持。
- **だが full-match 49.6%/pf_jh 154 で依然不変**。

## 重要な訂正: pf_jh の主因は threatened-reinforce ではなかった

診断: seed0 で Python の `threatened_candidates` が非空なのは **わずか 6 turn**。
→ 経過11 の「reinforce=300」は **followup/rear_guard も own-target として混入**して
いた誤分類。真の pf_jh 主因は **emit_followup_moves / emit_rear_guard_moves**
(self-play で自陣間に ship を送る後処理 phase) で、threatened-reinforce ではない。

reinforce port 自体は正しい (parity 維持) が発火機会が少ない (6 turn) ため full-match
に効かない。followup/rear_guard が次の本命。

## NEXT ACTION
1. emit_followup_moves (各 source の best secondary capture) を調査・port
2. emit_rear_guard_moves (後方 ship を前線へ ferry) を調査・port
3. これらが pf_jh の主因。port して full-match 向上を実測。劣化なし維持前提。

---

# 経過 14 (2026-06-03 ~05:55) — followup/rear_guard の構造を特定

- emit_followup_moves: 各 source の 2nd capture (main 後の残弾で別 enemy/neutral)。
  → own-target ではない (経過11/13 の誤分類を再訂正)。
- emit_rear_guard_moves: 後方自陣 ship を前線寄りの自陣 planet へ ferry (own→own)。
  → これが「own-target move」の正体。

## JAX 構造ギャップ
resolver は (1) 1 source=1 launch (out[src]<0) + (2) target=enemy/neutral のみ。
→ followup (同 source 2 発目) も rear_guard (own→own) も出せず pf_jh に寄与。

## NEXT
1. followup port: resolver で 1 source 2 launch 許可 (FOLLOWUP_MIN_SHIPS gate)。
2. rear_guard port: own→own ferry。
3. 各 port 後 full-match/pf_jh/tripwire 実測 (劣化なし維持)。

---

# 経過 15 (2026-06-03 ~06:15) — followup infra + pf_jh の真の正体

## resolver を 1 source 2 launch 対応に (followup infra)

- out[src] を count 化 (0/1/2)、2nd launch は available-spent>=FOLLOWUP_MIN_SHIPS(8) gate
  かつ capture-only。emission を per-source collapse → 全 fire を slot 詰めに変更。
- turn0 20/20 + tripwire 2/4 維持。だが full-match 49.6%/pf_jh 154 不変 = followup も
  発火せず。

## pf_jh の真の正体 (直接 dump で確定)

pf_jh turn の Python move を dump (t46/t71/t72):
- t46: src12 が **26 ships 全部** (available=26)、src24 が 20 全部。**JAX は hold**。
- これは followup でも reinforce でもなく **main capture**。Python は full-commit で撃つが
  JAX は同 target を reject。
→ **pf_jh の主因 = JAX が full-commitment capture を過剰 reject している**
  (followup/reinforce 不足ではない)。over-fire 修正 (plan_shot guards) が over-correct
  したか、need/value/opening_filter のいずれかが本物より厳しい。

## NEXT ACTION
1. t46 src12→target を直接トレースし、どの check (aim_ok/need/value/opening_filter) で
   JAX が reject するか特定 (経過9 と同じ手法を pf_jh 側に適用)。
2. 過剰 reject を修正。followup/reinforce infra は保持 (無害、将来効く)。
3. 劣化なし維持。

---

# 経過 16 (2026-06-03 ~06:35) — pf_jh 確定: 多source swarm 未実装

## t46 直接トレースで確定

- Python src12 が target6 (45 ships) に **send=26**、src24 が **同 target6** に send=20。
  26+20=46 = need。**multi-source 協調攻撃 (swarm / process_multi_source_mission)**。
- 単一 source では誰も need=46 を満たせない → JAX (single-source only) は両方 reject → hold。
- **pf_jh の確定主因 = 多source swarm 未実装**。followup/reinforce/過剰reject ではなかった
  (経過11/13/15 の仮説を最終訂正)。

## swarm port のスコープ (process_multi_source_mission, mission_resolver.py:84)

1 target に対し複数 source の option を束ね、turns 順 → -limit → src_id で sort し、
remaining = need を各 source に割当 (send = min(limit, max(0, remaining - 他src残)))。
remaining==0 になれば全 source 同時 launch。build_swarm_missions が pair/trio を生成。

## NEXT ACTION
1. swarm を port: target ごとに上位 source を集め need まで協調割当。
   固定shape (target × top-K source) で実装。
2. turn0 維持 + pf_jh 減 + full-match 向上を実測。tripwire 維持。
3. これが pf_jh の本丸 (154 の大半)。
