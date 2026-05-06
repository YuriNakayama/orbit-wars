# rulebase/case9 — anti_ping_pong (iter5 result, ACCUMULATE port)

> 作成日: 2026-05-05
> 関連: `iter5_plan.md` (state.md 統合)、`iter1-4_*.md`
> Status: **棄却** (42.5% / 200戦、iter2 比 -7pp、過去最低)

## サマリ

case7 から ACCUMULATE 関連 (config 19定数 + missions/stay.py 488行 + WorldModel cached_travel_time + strategy.py 配線) を 5 phase に分けて慎重に port。Phase 4 で
ACCUMULATE_ENABLED=True にして 200戦評価したところ **42.5%** (v9=85 / v4=115)。

iter1 (46%) すら下回り **過去最低水準**。途中で 1 度バグを発見・修正
(惑星 1 個でも ACCUMULATE が hold するため出撃不能 → `len(my_planets) > 8` ガード追加)
したが、修正後も全体で iter2 (49.5%) 比 -7pp の悪化。**ACCUMULATE port は case9 では逆効果**と結論。

## 数値

### Phase B: vs baseline_v4 200戦 (seed 7000-7199)

| 配置 | エピソード | v9 勝 | v9 勝率 |
|---|---|---|---|
| seat=0 (v9 先手) | 100 | 39 | **39.0%** |
| seat=1 (v9 後手) | 100 | 46 | **46.0%** |
| **合計** | **200** | **85** | **42.5%** |

- 平均試合長: 376.2 turn
- **Seat bias が逆転**: iter1-4 は seat0 (先手) 優位、iter5 は seat1 (後手) 優位。ACCUMULATE が「先手優位の中盤押し切り」を阻害している

### iter1–5 サマリ

| iter | 主要変更 | n | 勝率 | seat bias | 採否 |
|---|---|---|---|---|---|
| 1 | cooldown 抑止 (3,5,3) | 100 | 46.0% | 16pp | 棄却 |
| **2** | **bypass=8 + 値短縮 (1,2,1)** | **200** | **49.5% (best)** | **7pp** | **best** |
| 3 | bypass=10 緩和 | 180/200 | 47.8% | — | 棄却 |
| 4 | multi-source 拡張 | 200 | 47.0% | 14pp | 棄却 |
| **5** | **ACCUMULATE port** | **200** | **42.5%** | **-7pp (逆転)** | **棄却** |

### chunk 別累積勝率

| chunk | iter2 | iter5 |
|---|---|---|
| 0–60 | 56.7% | 41.7% |
| 0–100 | 53.0% | 39.0% |
| 0–140 | 52.9% | 41.4% |
| 0–180 | 51.1% | 43.3% |
| 0–200 | 49.5% | **42.5%** |

iter5 は最初から最後まで iter2 を **大きく下回る** 軌跡。後半若干改善するが採否しきい値には程遠い。

## 診断

**仮説 (ACCUMULATE で余剰 ship を遠距離 1 発 capture に転用) は不成立**。理由 (推察):

1. **case9 の ACCUMULATE は STAY_BURST と切り離して動かしている**。case7 では STAY_BURST (1ターン arbitrage) と ACCUMULATE (多ターン蓄積) が補完関係で機能する設計。STAY_BURST 配線を省いた case9 では「蓄積するけど短期の発火がない」状態 = ship が貯まる前に試合が動く
2. **ACCUMULATE_KNEE_SHIPS=60 は case9 の中盤シナリオには大きすぎる**。replay 分析で見た「t=100 で 100+ ships 大型 launch」は v9 勝ち試合で稀に起きるイベントで、60 ships で発火させても **既に case9 の通常 mission が 100+ ships を撃てる局面の前段 (惑星過多前) で hold が連発し攻めを止める**
3. **Seat bias の逆転**: iter1-4 で先手優位 (中央制圧で雪だるま) だったが、ACCUMULATE は中央制圧の launch を hold で止めるため先手優位が消える。後手は元々 launch 量が少ないので相対的影響が小さい

## 判定

**棄却**。`ACCUMULATE_ENABLED=False` に戻し case9 = iter2 等価に復帰済み。

## NEXT ACTION (iter6)

iter1-5 の累積で **iter2 が依然として best (49.5%)**。次の方向性:

1. **iter2 を 300 戦で再評価** — 49.5% (Wilson 95% CI [42.7%, 56.3%]) は信頼区間が広すぎ、真値が +5pp 達成している可能性が完全には否定できない。**評価コスト面では rust simulator なしには 300戦は時間がかかりすぎる**
2. **agent 速度最適化** — build_world / plan_shot の cache 化で turn 計算を高速化、200戦時間を半分に。**性能を下げない条件 (ablation で ±2pp)** で許容
3. **iter2 をベースに小規模 cooldown 値 tuning** — `PING_PONG_PAIR_COOLDOWN_TURNS: 1 → 2` (iter2 と iter1 の中間)、`HARASS_TARGET_COOLDOWN_TURNS: 2 → 3` などのスイートスポット探索
4. **ACCUMULATE port は完全捨てない** — STAY_BURST も同時に port すれば case7 同等の挙動になり case9 でも効く可能性。次は両方一緒に port する大型 iter で再挑戦
