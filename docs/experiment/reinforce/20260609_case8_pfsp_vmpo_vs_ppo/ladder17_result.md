# case8 ladder17 — strict 対戦量を大幅増 RESULT

> 関連: ladder16_result.md / hypotheses.md
> run_id: 20260615-084334__feature-poc-v-mpo__4a2c2c9__seed0 / commit: 4a2c2c9f
> case: reinforce_case8_vmpo_ladder17 / GPU: A100 80GB SECURE ($1.39/h)
> 開始: 2026-06-15T08:49Z / 早期停止: iter10 (量仮説の反証が明確、低速runコスト回避) / pod destroy済 (0確認)

## Summary

逆カリ路線を捨て、軽量T0ラダーに戻して strict 対戦量を増やす (mix_strict 0.6→0.85,
低T0偏重ladder [0,0,50,...], 素strict T0=0 を2枠) ことで「序盤を学習しない」を量で
押す狙い。結果、**素strict (T0=0) は高volume訓練でも win 0.0→0.01・reward
-3.44→-3.32 とほぼ動かず**、held-out strict_v1 も 0 のまま。量仮説は反証。
さらに2つの構造が明確化: (1) 素strict段の reward -3.44 は shaping (-2.4) が ±1勝利
信号を埋没 → 勾配が material 蓄積へ、(2) f_var が勝てる高T0段 (win 0.5-0.67) に集中
し勝てない T0=0段を 10iter中2回しか選ばない (難段過小サンプリング)。

## Numbers

### strict rung (f_var が T0 で勝率階層化)

| iter | T0 | win | reward |
|---|---|---|---|
| 0 | 50 | 0.01 | -2.91 |
| 1 | 0 | 0.00 | -3.44 |
| 2 | 225 | 0.667 | +1.24 |
| 3 | 200 | 0.542 | +0.62 |
| 5 | 0 | 0.01 | -3.32 |

- T0=0 rung: iter1=0.0 → iter5=0.01 (5iter訓練でほぼ不変)
- strict rung rollout mean = **411s** (軽量ladderでも 192ep×500手の strict on-device
  rollout が本質的に重い。期待した ~195s には戻らず)
- mix_strict 0.85 でほぼ全iterがstrict段 → 高volumeだが低速 (1h で iter10)

### held-out (turn 0)

| iter | strict_v1 | full |
|---|---|---|
| 0 | 0.0 | 0.734 |

## Diagnosis

campaign 17 ladder + 逆カリ3run + 量1run を通じ、ボトルネックが triangulate された:
- **curriculum 形状でも対戦量でもない**。T0/ε 窓、逆カリ warmup、高volume いずれも
  素strict (turn 0) を 0% から動かせない。
- **真の主因は reward 支配**: 素strict段で shaping (material比, -2.4) が終端±1勝利を
  埋没させ、勾配が「勝つ」でなく「material蓄積」に向く (ladder13 で terminal_scale=8
  を試すも過剰で棄却)。
- **副因は f_var の難段回避**: 勝率0付近の T0=0段は f_var 重み (x(1-x))^p が小さく
  ほぼ選ばれない → 難段に有効勾配が届かない。

## Decision

- 採否: **rejected (量仮説)**
- 次の一手 (要ユーザー確認): **reward 支配対処 + 難段強制サンプリング** の複合。
  (a) terminal_scale を gentle に (ladder13 の 8 は過剰 → 2-3) かつ shaping_coef を
  下げ「勝利信号優位」を素strictで確保、(b) force_rung_low_every / mix で T0=0段を
  毎K iter 強制照射し難段に勾配を届ける。reward と sampling は「素strict段で勝利へ
  勾配を向ける」単一の狙いの表裏。resume は ladder11 best.pt。

## Artifacts

- model: `data/output/models/reinforce/case8_vmpo_ladder17/runs/20260615-084334__feature-poc-v-mpo__4a2c2c9__seed0/` (best.pt, metrics.json iter 0-10)
