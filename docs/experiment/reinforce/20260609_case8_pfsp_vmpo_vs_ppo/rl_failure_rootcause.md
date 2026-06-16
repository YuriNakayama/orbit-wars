# なぜ RL が strict で失敗するか — コードレベル根本調査 (2026-06-16)

> 動機: 模倣学習は過去に失敗 (poc_bc: BC重み自体が rulebase に 0%)。「strict が強い」前提で
> 対策を重ねてきたが、ユーザー指摘により **RL の学習機構そのものの欠陥** を疑い再調査。

## 過去の BC 失敗 (本命にすべきでない裏付け)
- `poc_bc_result.md`: BC warm-start (Kaggle人間棋譜由来) は **BC重み自体が rulebase に 0%** で
  上昇軌跡なし → NO-GO。人間棋譜 BC は rulebase に traction が無い。strict-BC は別だが、
  「BC が銀の弾丸」という前提は過去に否定済。

## 決定的所見: RL機構の欠陥 (strict が強い、ではない)

### 観測された異常 (ladder18 metrics)
- self_snapshot 段: win 0.65, entropy ~37-44 (健全)
- **strict T0=0 段 (iter4): entropy が 44→11.84 に崩壊、policy_loss 33→11** ← policy 退化
- value_loss は全段 ~0.01 と極小 (= 「確実に負ける」を正しく予測 → advantage 分散ゼロ)

### コードレベルの根本原因 — degenerate update に対する guard が無い

**1. advantage 正規化が零分散バッチでノイズを増幅** (`vmpo_jax.py:257-258`)
```python
advantages = (advantages - mean) / (std + _EPS)
```
素strict段は agent が ~100% 負けるため、return がほぼ一様 (全敗) → `std(advantages) ≈ 0`。
これを `(std + EPS)` で割ると **数値ノイズが大きな正規化advantageに増幅される**。

**2. V-MPO top-half がノイズ選択を強化** (`vmpo_jax.py:167-188`)
```python
thresh = median(advantages); top_mask = adv >= thresh   # 上位50%
psi = softmax(adv/η over top half)
l_pi = -Σ psi·new_lp                                     # 重み付きMLE
```
全敗バッチでは「上位50%」= **最も運の良かった (ノイズ的) 負け試合**。L_π はその
ノイズ選択された行動を最尤で強化 → policy がノイズを模倣し **entropy 崩壊**。

**3. degenerate update を skip する guard が無い** (grep 確認: 該当なし)
win==0 / advantage分散≈0 のバッチでも通常通り勾配を流す → **素strict段が共有policyを
poison**。f_var ladder は1つのpolicyを self(勝てる)と strict(全敗)の両方で訓練するため、
strict段の退化勾配が self で得た学習を壊す destructive interference。

### no_op_bias=8.0 による over-fire の助長 (`model_jax.py:402`)
NO-OP logit に -8.0 → agent は「撃たない」を罰され**常時発射**。aim診断の「strictの3-4倍
fleet乱発・87%が戦力不足」と直結。これも勝てない一因 (無駄発射で戦力分散)。

## 結論: strict を倒せないのは「RLが degenerate batch で自壊するから」

「strict が戦略的に強い」のは事実だが、**それ以上に、~100%負けるバッチで RL が policy を
能動的に破壊している**。これが「探索で勝ちを見つけられない」より深い層の問題。zero-variance
は「学習しない」だけでなく「**正規化×top-half で退化勾配を生み policy を壊す**」。

## 真の修正候補 (BC より前にやるべき RL 機構の修正)

| 修正 | 内容 | 根拠 |
|---|---|---|
| **★A. degenerate-batch guard** | win_rate==0 (or adv分散<閾) の iter は **policy更新をskip / advを正規化しない** | 退化勾配の poison を断つ。最小・低リスク |
| **★B. advantage正規化の分散下限** | `std` に下限 (例 max(std, 0.1)) or 全敗時は正規化off | ノイズ増幅を止める |
| **★C. no_op_bias 見直し** | 8.0→0〜2 に下げ over-fire 抑制、または「撃つべき時だけ撃つ」を許容 | 無駄発射の戦力分散を是正 (aim診断) |
| D. strict段を別policyヘッド/分離更新 | self と strict の勾配干渉を分離 | destructive interference 回避 |

→ **A+B+C は RL の健全性修正で、BC より優先度が高く低リスク**。これらを入れた上で素strict
段の win/entropy が崩れなくなれば、handicap や弱め curriculum が初めて機能する素地ができる。

## 次の一手 (推奨)
ladder21 = **A (degenerate guard) + B (adv分散下限) + C (no_op_bias↓)** の RL健全性修正。
これは「strict を倒す新手法」でなく「**RL が自壊しないようにする**」修正で、過去の全 curriculum/
handicap 実験が effく前提条件。resume ladder18 best.pt。単一の狙い (degenerate update対策) の
複数実装なので A/B 上は1点の工夫。

## Sources (RL機構の一般論)
- 既存 campaign metrics (ladder18 entropy崩壊 iter4) + コード (vmpo_jax.py:167-258, model_jax.py:402)
- [Dealing with Sparse Reward (Daaboul)](https://medium.com/@m.k.daaboul/dealing-with-sparse-reward-environments-38c0489c844d) — 勝ちが無いと学習不能 (本件はさらに「自壊」する)
- 過去結果: poc_bc_result.md (BC失敗), strict_loss_replay_diag.md (over-fire/aim), ladder19/20_result.md (zero-variance)
