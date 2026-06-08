# parity 調査: H4 の vs-rulebase 0/30 の根本原因

時刻: 2026-06-08 / 「scale で勝った」が本物 rulebase に転移しなかった原因の事実確認。

## 検証した命題
「H4 の 0/30 は train(JAX)/eval(本物) parity ズレが原因で、parity を直せば JAX proxy は信頼できる」
— これが事実か実測で確認した。

## 段1: H4 ckpt を本物 rulebase で再評価 (相手違い仮説の棄却)
H4 の学習相手 `baseline_jax_full` は **baseline_v1 の JAX 近似**。評価を v8 でなく正しい v1 で:
- ckpt_i131 × 本物 **baseline_v1** (30戦) = **0/30**
- ckpt_i131 × 本物 **baseline_v8** (30戦) = **0/30**
→ 相手を正しく v1 にしても全敗。「v8 が不当に強かった」説は棄却。真の train/eval discrepancy。

## 段2: opponent action parity 実測 (★根本原因)
同一盤面 40 個で `baseline_jax_full`(JAX, 学習相手) vs `_host_python_v1_action`(本物 v1, 評価相手)
の action を比較 (`/tmp/parity_probe.py`):
- **exact_match = 4/40 (10%)** ← うち noop 盤面 4 件のみ。発火盤面では実質 **0% 一致**。
- **mean Jaccard = 0.100**。
- jax_fires=22 / py_fires=36 → **JAX が撃たない盤面で本物は撃つ**ことが多い。
- 撃つ盤面でも狙う planet は近いが **angle/ships が全く違う** (例 jax=(19,2,46) vs py=(19,-3,20),(19,-2,35))。
→ **`baseline_jax_full` と 本物 `baseline_v1` はほぼ別エージェント**。これが 0/30 の直接原因。

## 根本原因: baseline_jax_full は v1 の「近似」であり action-parity 非保証
- `baseline_jax_full` docstring: "baseline_v1 **feature** parity" (入力特徴の話、action ではない)、
  "Phase 6: 100-game eval vs baseline_v1, **weight tuning**" (勝率を合わせる調整であり action 一致ではない)。
- コード: 単一 243 行ファイルで `SCORE_PROD_WEIGHT=8.0` 等の固定重み、
  "We **approximate** this in a fixed-shape way" / "Keep lite parity **for now**" と明記。
- **action-identity test が存在しない** (`test_geometry_jax.py` のみ。rulebase JAX port の
  `test_agent_jax_identity.py` のような action 一致テストは無い)。
- 一方 本物 v1 は 23 ファイル・multi-mission strategy。jax_full はそれを fixed-shape JAX で**近似**したもの。

## 結論 (事実に基づく)
1. **parity ズレは実在する** (action 一致 10%)。これが H4 0/30 の直接原因。
2. ただし **`baseline_jax_full` は意図的な近似** であり「直すべき単純バグ」ではない。
   action-parity を取るには v1 の multi-mission strategy を fixed-shape JAX で**完全再実装**する必要があり、
   これは memory `project_rulebase_jax_parity_failure_mode` が扱う難問 (float32 tie-break 発散等)。
3. **「parity さえ直せば JAX=Python」は未検証**: 段3 (agent ckpt の JAX↔torch parity) は未実施。
   仮に opponent parity を完全に取っても、featurizer/agent 側に別の差が残る可能性は排除できていない。

## 段4: parity ズレは env か agent か (切り分け実測)
`state_to_obs(jax_state)` の faithfulness と agent-layer の差を分離 (`/tmp/env_agent_split.py`):
- **ENV層は健全**: `obs_planets == state_planets` 全行一致 (s0:32=32, s1:20=20, ...)。state_to_obs は正しい。
- **差分は AGENT層 (JAX rule 実装)**:
  - source-planet 一致 19/30 (63%) — 狙う planet が半分強しか合わない。
  - fire-rate: JAX rule 16/30 vs Python 27/30 — **JAX rule は保守的で撃たない盤面が多い**。
  - shared launch = 0 (JAX 16発 / Python 56発で重複ゼロ) — 同 source でも angle/ships が全て違う。
→ **parity ズレは env(シミュレータ)でなく agent(JAX rule)由来**。2層の差: ①target選択 ②aim/allocation。

## どの JAX rule も parity-exact でない (test 状況)
- **action 100% 一致 test は case1 にのみ存在**、対象は `baseline_jax/core_jax/agent_full_jax` (1:1狙い)。
  だが **その test は FAIL** (5/5 seed, `test_jax_port_action_equivalence_over_selfplay`)。
- case2-9 の `test_agent_jax_identity` は "jax_wins>=3" の弱い勝率 test のみ (action 一致не検証)。
- baseline_jax_full には action-parity test 自体が無い。
→ **リポジトリ内に「本物と action 完全一致」が検証された JAX rule は存在しない** (memory rulebase_jax_parity_failure_mode の float32 tie-break 発散)。

## 段5: 各 JAX rule の parity 実測 — case8 はほぼ parity-exact ★
同手法で case8 JAX rule (`build_world_features_from_state`→`compute_actions`) vs 本物 v8 を実測
(`/tmp/case8_parity.py`):

| 指標 | case1 baseline_jax_full vs v1 (H4が使用) | **case8 baseline_jax vs v8** |
|---|---|---|
| full exact | ~10% | **90% (27/30)** |
| source match | 63% | **100% (30/30)** |
| fire-rate (JAX/Py) | 16/27 (保守的すぎ) | **27/27 (一致)** |
| shared launches | 0/56 | **52/56** |

s1-s3 はサンプルレベルで完全一致 (`jax=[(15,3,19)] py=[(15,3,19)]`)。残り ~10% は float32
tie-break edge (memory `rulebase_jax_parity_failure_mode`) と推定。
→ **case8 baseline_jax は本物 v8 のほぼ忠実な port (~90%)**。H4 が使った baseline_jax_full
(v1 の ~10% 近似) とは雲泥の差。**(A) の学習相手は case8 が正解**。

## 結論 (段1-5 を総合)
- H4 0/30 の原因は **学習相手 baseline_jax_full が本物 v1 の粗い近似 (action 10%)** だったこと。env は健全、agent rule の差。
- **解決策 = 学習相手を parity-exact (~90%) な case8 baseline_jax に替える**。これで train/eval gap が大幅に縮む。
- ただし「parity 良い相手なら本物に勝てる agent が学習できる」は **未だ実証前** — case8 を相手に学習し直して本物 v8 で評価する実験 (次) で確認する。

## 正しい次の選択肢 (どれも「scale/機構」ではない)
- **(A) 学習相手を action-parity 保証済みの相手にする**: case8 の JAX port は parity-exact
  (`test_agent_jax_identity` 通過)。`baseline_jax_case8` を学習相手にすれば train/eval gap が縮む。
- **(B) 本物 rulebase を host_callback で学習相手に混ぜる** (`python_v1` mode は既存)。rollout 重いが本物経験。
- **(C) baseline_jax_full に action-parity test を追加**し、v1 と一致するまで修正 (難問、要工数)。

## 実測コマンド
- eval: `python -m pipeline.reinforce.case7.evaluation.eval_ckpt_vs_rulebase --ckpt ... --baseline baseline_v1 -n 30`
- parity probe: `/tmp/parity_probe.py` (要 bot/ cwd, PYTHONPATH=.)
