# case8 ladder16 — 逆カリキュラム 強制後退 (warmup advance 単一compile化) RESULT

> 関連: ladder15_result.md / hypotheses.md
> run_id: 20260615-055642__feature-poc-v-mpo__adc070e__seed0 / commit: adc070ef
> case: reinforce_case8_vmpo_ladder16 / GPU: A100 80GB SECURE ($1.39/h)
> 開始: 2026-06-15T06:03Z / 早期停止: iter22 (実行コスト律速確定 + 序盤で勝率低下を確認) / pod destroy済 (0確認)

## Summary

ladder15 の ~870s/strict段を「再compile が原因」と見て warmup advance を traced
scalar + static-max scan で単一compile化したが、**strict段は依然 ~867-872s で不変**
→ コストは compile でなく **実行** (strict をオンデバイスで相手500手 + warmup自己
対戦最大320手 実行する構造的コスト) だったことが確定。さらに warmup が序盤へ降下
するほど strict段勝率が **低下** (225→0.359, 125→0.161, 100→0.161) — 逆カリの後退は
agent を strict の難しい序盤に晒すが、**agent は序盤を勝てるように学習せず、ただ
負けが増える**。逆カリ路線 (ladder14-16) は棄却。

## Numbers

### strict 段 rollout 時間 (単一compile化でも不変 = 実行律速)

| iter | rc_warmup | rollout | win |
|---|---|---|---|
| 1 | 275 | 944.6s (compile込) | 0.307 |
| 4 | 250 | 871.0s | 0.339 |
| 9 | 225 | 868.6s | 0.359 |
| 10 | 200 | 867.1s | 0.312 |
| 17 | 175 | 872.6s | 0.281 |
| 18 | 150 | 872.2s | 0.286 |
| 19 | 125 | 866.4s | **0.161** |
| 21 | 100 | 870.4s | **0.161** |

- iter4 (cached graph) = 871s ≈ iter1 (944s − 一回compile) → **再compileは無く実行が律速**
- self_snapshot 段は ~19s。strict 段だけ ~870s = strict オンデバイス実行コスト

### 決定的所見: warmup 降下に伴う勝率 *低下*

warmup 300→100 で win 0.36→0.16。**序盤に近いほど agent は勝てない** = strict の
決定的序盤を学習できていないことの直接証拠 (ladder15 の用量反応を後退方向でも裏付け)。

## Diagnosis

2点が確定した:
1. **逆カリの実装コストは構造的** — strict-self warmup advance + strict 相手の
   オンデバイス実行が ~870s/段。compile 最適化では解けない。warmup を片seat化/
   事前計算しても、strict 相手の 500手 rollout 自体が重い。
2. **逆カリの仮説は支持されない** — 後退で序盤に晒しても agent の序盤勝率は上がらず
   下がる。「序盤を経験させれば学習する」前提が、少なくとも現在の経験量/reward では
   成立しない。core 問題は「序盤を学習できない」こと自体。

## Decision

- 採否: **rejected (逆カリ路線 ladder14-16 を打ち切り)**
- 次の一手: **ladder17 = strict 対戦量を大幅増** (ユーザー選択)。重い warmup advance
  を捨て軽量 T0ラダー (~195s/段) に戻し、mix_strict 0.6→0.85 + 低T0偏重ladder
  [0,0,50,110,140,170,200,225] で素strict序盤の対戦量を3-4倍に。「序盤を学習しない」
  を量で押す (絶対経験量2-3桁不足という phaseB 診断への直接対処)。resume ladder11
  best.pt。

## Artifacts

- model: `data/output/models/reinforce/case8_vmpo_ladder16/runs/20260615-055642__feature-poc-v-mpo__adc070e__seed0/best_i3_win0.7031.pt`
- metrics: 同 dir / metrics.json (iter 0-22)
