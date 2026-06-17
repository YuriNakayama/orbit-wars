# case7 「ルールベースに勝つ」ループ — iter09 RESULT

時刻: 2026-06-03 10:46 (cron tick 13)

## やったこと: memory features 修正 (研究駆動の学習ロジック改善)
Generals.io 論文 (arXiv 2507.06825) が加速要因とした memory features の欠落を修正。
学習 rollout の `update_history_jax` に **空でなく agent の実 launch** (env_actions[seat]
の from_pid/ships/valid) を渡すよう変更 (ruff clean, smoke pass)。
BC warm-start + ratio shaping + 3段 curriculum、14 iter。

## 結果
| model | vs baseline_v1 (10戦) |
|---|---|
| iter09 (memory features) | **0/10** |

- vs self_snapshot は iter08 (mem無) と**ほぼ同等** (0.67-0.83) = memory features は
  学習ダイナミクスを動かさず。
- vs lite も 0.167 のまま。research が「memory features **＋** GPU scale」両方を
  要したのと整合 (memory features 単体では不十分)。

## ★ループ全体の結論 (13 tick・10+ variant、確定)
| lever | vs baseline_v1 |
|---|---|
| BCなし / 生BC / BC-RL / 3段 / self-play30 / 本物v1直接 / production shaping / memory features | **全て 0/10** |
| il_v11 (最強 imitation) | 0/10 |

- RL machinery 健全 (kl>0, vs self reward+, beats random 10/10)。
- 変換/featurizer parity 81件 pass。
- 敗因 = production gap (score 51 vs 13000-16000)。
- **どの小規模ローカル改善でも天井を破れない。research が示す通り根因は scale**
  (Generals.io: 同レシピで H100×36h)。**ローカル小規模 RL では baseline_v1 に勝てない**
  ことが網羅的に確定。

## 価値ある成果物 (本ループの産物)
- case7 family (PFSP pool × ratio shaping)、resume 機能、incremental metrics、
  memory features 修正、評価/変換パイプライン、iter01-09 の系統的 ablation 記録。
- これらは GPU scale や別アプローチの土台として再利用可能。

## NEXT ACTION
- ローカル小規模の探索空間は尽きた。残る唯一の道は **GPU scale** (research 準拠) だが
  push (remote 公開) + 課金 + memory 的に v1 勝利は不確実。
- → ユーザーに **GPU scale で続けるか / 別 family (rulebase) に振るか / ここで一区切りか**
  を明示的に問う (重い一歩は無断実行しない)。
- memory features 修正は有益なので commit して保全。
