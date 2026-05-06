# rulebase/case9 — anti_ping_pong (iter9 result, 300-game confirm of HARASS=3)

> 作成日: 2026-05-06
> 関連: `iter8_result.md`, `iter9_state.md`, `iter1-7_*.md`
> Status: **棄却** (300戦 48.3%、+5pp 未達、iter8 200戦は seed 偶発と確定)

## サマリ

iter8 で 200戦 55.5% (採択) を 300戦で confirm したところ **48.3%** (v9=145 / v4=155)。
iter8 比 **-7.2pp** の大幅低下、信頼区間 [42.7%, 53.9%] 全域が +5pp 未達。
**iter8 の採択は seed 10000-10199 の運によるもの** と確定。HARASS_MIN_TARGET_PRODUCTION=3
は実質改善要因ではない。

## 数値

### Phase B: vs baseline_v4 300戦 (seed 11000-11299)

| 配置 | エピソード | v9 勝 | v9 勝率 |
|---|---|---|---|
| seat=0 | 150 | 79 | **52.7%** |
| seat=1 | 150 | 66 | **44.0%** |
| **合計** | **300** | **145** | **48.3%** |

- Wilson 95% CI: 約 [42.7%, 53.9%]
- 平均試合長: 373.7 turn (iter2 / iter7 等価)
- Seat bias: 8.7pp (iter7 5.4pp、iter8 7pp より悪化)

### chunk 別累積勝率の推移 (iter9)

| chunk | 累積勝率 |
|---|---|
| 0–60 | 56.7% |
| 0–100 | **57.0% (peak)** |
| 0–160 | 53.1% |
| 0–200 | 51.0% |
| 0–240 | 49.2% |
| 0–280 | 48.6% |
| **0–300** | **48.3%** |

→ 序盤 100戦 57% peak は iter7 と完全に一致するパターン。**100戦時点の +5pp 達成は再現性なし、後半で必ず 50% 帯に回帰**。

### iter8 (200戦 55.5%) vs iter9 (300戦 48.3%) の seed 依存

| Run | seed range | n | 勝率 |
|---|---|---|---|
| iter8 | 10000-10199 | 200 | **55.5%** |
| iter9 | 11000-11299 | 300 | **48.3%** |

→ HARASS=3 設定は **iter8 で +5pp 達成、iter9 で +5pp 未達** = 真値が 50% 付近で seed range の運に左右されている。+5pp は実質 **seed 偶発の範囲外**。

## 診断

iter1-9 を通じて確認:
1. **case9 設計の真値**: vs baseline_v4 で **~50%** (case4 同等)
2. **+5pp 達成の seed 偶発性**: 200戦で 55%+ になることが偶発的にあるが (iter8)、300戦に伸ばすと 50% 帯に回帰 (iter9)
3. **HARASS_MIN_TARGET_PRODUCTION=3 は中立改善**: 一部の試合で機能する場面はあるが全体としては誤差レベル
4. **完了条件 60% は遥かに遠い**: 真値 50% 付近では 60% 達成は信頼区間外

## 判定

**棄却 + HARASS_MIN_TARGET_PRODUCTION を 2 に戻す**:
- iter8 の 200戦 55.5% は seed 偶発と確定
- iter9 の 300戦 48.3% が真の評価
- case4 default (=2) を維持する方が安全 (case4 自体が tuned 値)

case9 = iter2 best 設計 + iter6 plan_shot cache に戻す (iter6 まで採用された設計)。

## NEXT ACTION (iter10)

iter1-9 で **cooldown / 閾値系の小修正では +5pp 不可** が確定。残る選択肢:

1. **大型変更**: ACCUMULATE+STAY 同時 port (5+ 周回かかる、iter5 単独失敗の教訓を踏まえ STAY も配線)
2. **case10 で別アプローチ**: case9 を anti-ping-pong 機構の保管としてキープし、case10 で全く異なる戦術
3. **Loop 終了**: 9 iter 投下で +5pp 達成不可、loop の探索コストに見合わない判断

## 成果物

- 評価ログ: `/tmp/compare_v4_iter9.log` (300戦 summary)
- iter6 plan_shot cache は引き続き有効
