# case8 strict JAX port を他 case へ展開する計画

時刻: 2026-06-09。case8 (action 90-100% 一致の full-JAX port, grid+allocator, ~5400行/14
module) を他 case に展開する依頼への調査 + 実装記録。

## 前提 (Explore 調査で確定)

case8/baseline_jax は **~5400行・14 module** の高忠実 port (元々 ~60-80h で構築)。各 case の
Python 戦略は case8 と差分があり、**機械的な並列コピーでは不可**。差分クラス:

| case | 差分クラス | 工数 | 差分の中身 |
|------|-----------|------|-----------|
| **case4** | **cache のみ** | **低 (済)** | config/strategy/world_model **byte一致**。コピーで parity 保証 |
| case9 | dispatch history | 中 | + ANTI_PING_PONG (cross-turn 状態あり、下記) |
| case6 | STAY burst | 中 | + 新 mission + strategy signature 変更 |
| case2 | physics regression | 高 | physics.py が case8 より ~200行欠落、要差し替え |
| case3 | rollout | 高 | 別 decomposition + rollout feature |
| case7 | STAY+ACCUMULATE | 高 | multi-turn state machine (STAY/ACCUMULATE) |
| case1 | planner arch | (archive) | 3/7 mission のみ、別 planner 構造 |

## 移植テンプレ (case4 で確立)

case8 と Python ロジックが byte 一致なら、`case8/baseline_jax/*.py` の strict module 11 本
(agent_jax, aim_adapter, aim_jax, allocator_jax, geometry_jax, missions_capture_jax,
physics_jax, plan_shot_jax, scoring_jax, timeline_jax, world_features) を **コピーするだけ**。
全 import が相対 (`.`) + `orbit_wars_jax` のみで case8 固有の絶対 import なし。config 定数は
JAX module 内に bake されているため、config 一致なら定数差し替え不要。

検証: `compute_actions_jit(build_world_features_from_state(st,seat), modes)` vs Python agent を
12-30 state で full-exact 比較。**case4 = 12/12 (100%)**。registry に `jax_v4` 登録。

## case8 自身の残差 (90% の正体, 2026-06-09 実測)

30 state で full-exact 27/30。残 3 state の中身:
- s13/s27: **ship 数一致**、angle のみ ~1.5° 差 (aim solver の tie)。
- s21: 同一 4 launch、ship 配分が 2 target 間で swap (allocator ordering tie)。
→ 残差は **angle tie + allocator 順序**であり ship 数 miscount ではない。core_jax port
(exact 50-75%, 残差=followup ship 数) より遥かに軽微。case4 は cache path がないため 100%。

## 各 case の実装スコープ (case4 以外)

### case9 (中工数): case8 base + ANTI_PING_PONG mask
- 状態: module-global `_DISPATCH_HISTORY: dict[(src_id,tgt_id)->step]`。agent() が毎 turn
  plan 後に記録、PING_PONG_PAIR_COOLDOWN_TURNS+2 で prune、step 後退で reset。
- hook: reinforce mission (`(src,dst)` cooldown=1turn で `continue`)、harass mission
  (target への直近 dispatch が cooldown=2turn 内なら `continue`)。my_planets<=8 で bypass。
- = **filter** (再スコアでなく候補除外)。
- JAX 化方針: `recent_dispatches: (P,P) int array (last step, 未発火=-inf)` を
  compute_actions の入力に追加 → harass grid と reinforce path で
  `step - hist[src,dst] < cooldown` の mask を valid に AND。host (agent wrapper) が
  JAX 返値から history array を毎 turn 更新。allocator scan 自体は不変。
- 注: case8 strict port は現状 single+snipe+harass+followup のみ wired。reinforce grid は
  allocator に KIND_REINFORCE はあるが grid builder 未配線 → case9 で reinforce cooldown を
  正しく出すには reinforce grid 配線も要 (case8 base の未完部分)。

### case6 (中工数): case8 base + STAY burst mission
- config +33 STAY_* 定数。新 mission stay.py + strategy signature 変更。
- 直交する新 family として candidate table に STAY 種別追加 + allocator 合流。

### case2 (高工数): physics.py が case8 版に未到達 (~200行欠落)。physics 差し替え + strategy 再検証。
### case3 (高工数): rollout decomposition。case8 baseline parity 後に rollout 別 feature。
### case7 (高工数): STAY+ACCUMULATE multi-turn state machine。config +67 定数。最難。
### case1 (archive): 3/7 mission + 別 planner 構造。strict port 対象外。

### case9 base の重要な実測 (2026-06-09)

case8 strict base を case9 にコピーし、**masked なし**で case9 Python と比較 → **12/12 full-exact**。
理由: ANTI_PING_PONG は cross-turn の dispatch history で発火するが、**単発 state 評価では history
が空のため cooldown が一度も発火しない**。つまり:
- **stateless 単発 / eval 用途**: base だけで parity-exact → `jax_v9` 登録可 (caveat 付き)。
- **実 multi-turn 対戦**: cooldown が発火し base は case9 Python と乖離する → history array 入力 +
  host 側 update の実装が必須 (未実装)。

→ `jax_v9` は「単発 parity exact、cross-turn ANTI_PING_PONG 未配線」として登録。GPU self-play で
opponent に使う場合、ping-pong 抑制の差は出る (case9 の核心 feature がまだ効かない) ことに注意。

## 進捗

- [x] case4: copy + parity 12/12 + registry (jax_v4) + commit (50a0d485)
- [x] case9: case8 base copy + 単発 parity 12/12 + registry (jax_v9, caveat 付き)。
      残: ANTI_PING_PONG history array 入力 + harass/reinforce mask + host update + multi-turn parity。
- [x] case6: case8 base + stay_jax.burst_held_mask + parity 12/12 + registry (jax_v6) + commit (a550fcb6)。
      base 11/12 → STAY burst-hold mask 適用で 12/12 (s1 planet15 hold が一致)。
      ★ハマり: world_features に builder が 2 つ (build_world_features=obs / build_world_features_from_state=JAX state)。
      両方に burst-hold を適用しないと parity probe (from_state 使用) で効かない。
      残: STAY_BURST_MAX_HOLD_TURNS cap (cross-turn) は host 側 (3連続 hold 上限)。
- [ ] case2/3/7: 個別構造移植 (高工数、各 parity 検証必須)
- [ ] case1: archive (別 planner 構造、strict port 対象外)

## strict port 進捗サマリ (2026-06-09)

| case | strict port | parity | registry | 戦略の個性 |
|------|:--:|:--:|:--|:--|
| case4 | ✅ | 12/12 (100%) | jax_v4 | case8 と同一 (cache のみ差) |
| case8 | ✅ (既存) | 27/30 (90%) | jax_v8 | base 戦略 |
| case9 | ✅ base | 単発 12/12 | jax_v9 | +ANTI_PING_PONG (未配線) |
| case6 | ✅ | 12/12 (100%) | jax_v6 | **+STAY burst-hold (配線済)** ← 初の別戦略 |
| case2/3/7 | ❌ | — | — | 高工数 (physics/rollout/ACCUMULATE) |

→ GPU opponent pool に **戦略バリエーション** (case6 の STAY) が初めて追加された。
