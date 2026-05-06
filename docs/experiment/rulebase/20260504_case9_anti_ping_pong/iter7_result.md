# rulebase/case9 — anti_ping_pong (iter7 result, 300-game confirm)

> 作成日: 2026-05-06
> 関連: `iter7_plan.md`, `iter6_result.md`, `iter1-5_*.md`
> Status: **棄却** (52.0% / 300戦、+5pp 未達、信頼区間上限ぎりぎり)

## サマリ

iter2 best 設計 + iter6 plan_shot cache の組み合わせを **300戦** で再評価。
**52.0% (v9=156 / v4=144、Wilson 95% CI [46.4%, 57.5%])**。
信頼区間上限 57.5% は採択しきい値 55% を超えるが、期待値 52.0% が未達のため
**iter2 ベース設計では +5pp 達成は確実ではない** と確定。

iter1-6 の累積で iter2 が best (49.5% / 200戦)、iter7 で 300戦 52.0% に上振れ
(seed 9000 range が iter2 seed 3000 range より好相性) したが、+5pp ボーダーは
seed range の運に左右されるレベル。

## 数値

### Phase B: vs baseline_v4 300戦 (seed 9000-9299)

| 配置 | エピソード | v9 勝 | v9 勝率 |
|---|---|---|---|
| seat=0 (v9 先手) | 150 | 82 | **54.7%** |
| seat=1 (v9 後手) | 150 | 74 | **49.3%** |
| **合計** | **300** | **156** | **52.0%** |

- **Wilson 95% CI: [46.4%, 57.5%]**
- 平均試合長: 369.2 turn (iter2 等価)
- Seat bias: 5.4pp (iter2 7pp / iter6 6pp より改善、cache 副作用は無し or 微改善)

### chunk 別累積勝率の推移

| chunk | 累積勝率 |
|---|---|
| 0–60 | 56.7% (peak 帯) |
| 0–100 | 57.0% (peak) |
| 0–120 | 58.3% |
| 0–140 | 55.0% |
| 0–160 | 53.8% |
| 0–200 | 53.0% |
| 0–240 | 52.9% |
| 0–280 | 53.2% |
| **0–300** | **52.0%** |

→ 序盤 100戦は iter2 比 +7pp 高い軌跡だったが、seed 9100-9299 で揺り戻し。
**真値は 50–53% 付近** と推察、iter2 (200戦 49.5%) と統計的同等で **+5pp 達成は seed 偶発に依存**。

### iter1–7 サマリ

| iter | 主要変更 | n | 勝率 | CI 95% |
|---|---|---|---|---|
| 1 | cooldown 抑止 | 100 | 46.0% | [36.4%, 55.7%] |
| 2 | **bypass=8 + 値短縮** | 200 | 49.5% | [42.7%, 56.3%] |
| 3 | bypass=10 | 180/200 | 47.8% | — |
| 4 | multi-source | 200 | 47.0% | [40.2%, 53.8%] |
| 5 | ACCUMULATE port | 200 | 42.5% | [35.8%, 49.3%] |
| 6 | plan_shot cache | 200 | 49.0% | [42.2%, 55.8%] |
| **7** | **iter2 設計 300戦** | **300** | **52.0%** | **[46.4%, 57.5%]** |

## 診断

**仮説結論**: iter2 の anti-ping-pong 設計 (bypass=8 + cooldown 短縮 + REINFORCE_MIN_DEFICIT=1) は **vs baseline_v4 で 50-52% を真値とする設計**。これは case4 と同等水準であり、ping-pong 抑制効果は確認できる (iter2 結果の launches -17.7%、ping-pong 件数 -11.5%) が、**勝率改善は誤差レベル**。

理由 (推察、replay analysis から):
- **case4 自体が高水準**: vs case3 で 70.3%、vs case2 で +5pp 以上。case4 ベースで cooldown 系の小修正だけでは突き抜けにくい
- **試合の決着が score 比較**: 80% の試合が 500 turn 完走で score 比較。中盤の領土拡大が雪だるま式に拡大、anti-ping-pong は中盤の戦術的詰めにしか効かない
- **ACCUMULATE 単独 port で逆効果 (iter5 -7pp)**: 余剰 ship 流用は STAY_BURST と組み合わせる必要があり、case7 全 port は 1-2 周回では完結不可

## 判定

**棄却**。iter1-7 を通じて iter2 設計が best (今回 300戦で 52.0%) と確定したが
+5pp 達成は不可。**iter2 設計を case9 の最終形態として固定** (現状維持)。

## NEXT ACTION (iter8)

iter2 ベースの cooldown tuning では限界 → **抜本的な設計変更**:

1. **case7 ACCUMULATE + STAY_BURST 同時 port** (大型 iter、5+ 周回): iter5 の失敗
   原因が「ACCUMULATE 単独で蓄積するけど短期発火がない」だった。STAY_BURST も
   配線すれば case7 同等の挙動 (case7 自体は v4 比でやや弱いが port での挙動を見る価値あり)
2. **case4 大改造**: capture mission の score weight 強化、production が高い neutral を優先取得
3. **Loop 終了**: case9 の探索は十分 (7 iter)、iter2 設計を case9 最終形として確定し、case10 を新規 case で別アプローチ

## 成果物

- 評価ログ: `/tmp/compare_v4_iter7.log` (300戦 summary)
- iter6 plan_shot cache が引き続き有効、副作用は seat bias を僅かに改善 (-1pp)
