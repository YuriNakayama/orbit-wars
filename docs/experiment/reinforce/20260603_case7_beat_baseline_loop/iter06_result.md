# case7 「ルールベースに勝つ」ループ — iter06 RESULT

時刻: 2026-06-03 04:36 (cron tick 8 続き)

## ユーザー判断
tick 8 で「GPU / imitation 底上げ / ゴール再設定 / 停止」を質問 → **「ループ継続」**。
方法指定なしのため、最も情報価値の高い切り分けを自走実施。

## やったこと: 学習モデル一族の天井を確認
出発点 (case9 模倣) が弱いのが根、ならより強い模倣なら勝てるのでは? を検証。
**imitation/case11 (il_v11、ローカル最強の per-planet 模倣) を直接 10戦 vs baseline_v1**。

## ★結果
| model | vs baseline_v1 (10戦) |
|---|---|
| il_v11 (最強 imitation) | **0/10** |

→ case9 だけでなく **最強の imitation も 0/10**。memory `imitation/case1 0/100` と整合。
**per-planet Set Transformer 学習モデル一族は丸ごと baseline_v1 に勝てていない**。

## 統合結論 (8 tick・7 variant)
| approach | vs v1 |
|---|---|
| case7 RL (BCなし16) | 0/10 |
| 生 BC (case9) | 0/10 |
| BC-RL 14 | 0/10 |
| 3段 curriculum 16 | 0/10 |
| self-play 30 | 0/10 |
| **本物 v1 直接学習** | **0勝** (reward -2.0) |
| **il_v11 最強 imitation** | **0/10** |

- parity test 81件 pass = 変換/featurizer は健全。0/10 は本物の弱さ。
- **学習モデル(この architecture)では小規模でも本物 v1 に勝てない**ことが確定。
  imitation 底上げも同族なので不可。

## 残る現実的な道 (ユーザー判断待ち、無断実行しない)
1. **GPU 大規模 RL**: memory で case6 は GPU でも本物 v1 に 0/10。リターン不確実 + 課金。
2. **別 family**: rulebase/case8 は既に v1 と互角〜上 (memory: 50.5%)。
   「v1 に勝つ」goal なら rulebase 改善が確実。case_jax (本物 parity rule) も候補。
3. **architecture 変更**: per-planet Set Transformer 以外 (例: rule を candidate 生成に
   組み込む hybrid)。研究レベルの工数。

## NEXT (loop 継続方針)
- これ以上「同じ小規模 RL を回す」のは 0/10 が確定しており**無駄打ち** → 自動学習は止め、
  loop は **状況維持 + ユーザーの方針判断待ち**とする。
- 次 tick: 本結論を要約しユーザーに方針 (GPU課金 / 別family / goal再設定) を再度仰ぐ。
  ユーザーが具体策を示すまで新規の課金/大方針転換はしない。
