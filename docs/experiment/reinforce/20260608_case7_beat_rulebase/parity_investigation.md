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

## 正しい次の選択肢 (どれも「scale/機構」ではない)
- **(A) 学習相手を action-parity 保証済みの相手にする**: case8 の JAX port は parity-exact
  (`test_agent_jax_identity` 通過)。`baseline_jax_case8` を学習相手にすれば train/eval gap が縮む。
- **(B) 本物 rulebase を host_callback で学習相手に混ぜる** (`python_v1` mode は既存)。rollout 重いが本物経験。
- **(C) baseline_jax_full に action-parity test を追加**し、v1 と一致するまで修正 (難問、要工数)。

## 実測コマンド
- eval: `python -m pipeline.reinforce.case7.evaluation.eval_ckpt_vs_rulebase --ckpt ... --baseline baseline_v1 -n 30`
- parity probe: `/tmp/parity_probe.py` (要 bot/ cwd, PYTHONPATH=.)
