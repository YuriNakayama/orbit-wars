# rulebase/case9 — anti_ping_pong (iter4 result)

> 作成日: 2026-05-05
> 関連: `iter4_plan.md`, `iter1-3_*.md`, replay analysis `data/output/experiment/rulebase/case9/replay_analysis/20260505_0700/`
> Status: **棄却** (47.0% / 200戦、iter2 比 -2.5pp)

## サマリ

直近 replay 分析 (`win 試合は t=100 で 100+ ships の大型 launch 1 撃成功」) を踏まえ、`MULTI_SOURCE_TOP_K=5→8` + `THREE_SOURCE_PLAN_PENALTY=0.75→0.85` で multi-source swarm を出やすくする 2 行変更を実施。
**200戦 47.0%** (iter2 49.5% 比 -2.5pp、iter3 中断 47.8% と同水準)。
中盤 100戦時点では **54.0%** で iter2 (53.0%) を上回ったが、後半 100戦で v9=40/100 (40%) と大きく失速。**seat=1 (後手) のみで 14pp 勝率が悪化** (54% vs 40%)。multi-source 拡張は **先手では効いたが後手では逆効果**。

## 数値

### Phase B: vs baseline_v4 200戦 (seed 5000-5199)

| 配置 | エピソード | v9 勝 | v9 勝率 |
|---|---|---|---|
| seat=0 | 100 | 54 | **54.0%** |
| seat=1 | 100 | 40 | **40.0%** |
| **合計** | **200** | **94** | **47.0%** |

- 平均試合長: 382.5 turn (iter2 369、iter3 中断より 13 turn 長い)
- Seat bias: **14pp** (iter2 7pp、iter3 中断時不明、iter1 16pp)

### iter1–4 比較

| iter | 主要変更 | n | 勝率 | seat bias |
|---|---|---|---|---|
| 1 | cooldown 抑止 (3,5,3) | 100 | 46.0% | 16pp |
| **2** | **bypass=8 + 値短縮 (1,2,1)** | **200** | **49.5% (best)** | **7pp** |
| 3 | bypass=10 緩和 | 180中断 | 47.8% | — |
| 4 | multi-source 拡張 (TOP_K=8, THREE=0.85) | 200 | 47.0% | 14pp |

### chunk 別累積勝率

| chunk | iter2 | iter3 | iter4 |
|---|---|---|---|
| 0–60 | 56.7% | 50.0% | 50.0% |
| 0–100 | 53.0% | 48.0% | **54.0%** |
| 0–140 | 52.9% | 46.4% | 47.1% |
| 0–180 | 51.1% | 47.8% | 47.8% |
| 0–200 | 49.5% | — | **47.0%** |

iter4 は 100戦時点までは過去最高 (54%) だが、**100→200 戦で v9 が 40 勝/100戦の超不調** に陥り全体平均を押し下げた。

## 診断

**仮説 1 部分支持 / 部分反証**:
- **支持**: seat=0 で 54.0% は iter2 同 seat (53.0%) を上回る → multi-source 拡張は **大型 launch 促進に貢献している可能性**
- **反証**: seat=1 で 40.0% は iter1 (-6pp 程度) より悪い → 後手では multi-source 過多が **defense 手薄を呼び弱体化**

**Seat bias 14pp の意味**:
- 後手は序盤の中立確保で先手より遅れる傾向 (Orbit Wars の構造)
- multi-source swarm は **「複数 src を同時投入」する性質上、defense 用 ship を吐き出しやすい**
- 後手で序盤遅れがある状態で multi-source を発火しすぎると、自惑星防衛が崩壊しやすい
- これは想定リスク (plan.md「swarm 過多で reinforce/defense が手薄」) が現実化したパターン

## 判定

**棄却**。iter5 で:
1. **multi-source 設定を元に戻す** (`MULTI_SOURCE_TOP_K: 8 → 5`、`THREE_SOURCE_PLAN_PENALTY: 0.85 → 0.75`)
2. **iter2 設計をベースに ACCUMULATE port を進める** (case7 stay.py 系の本格移植)
3. iter4 の **seat=0 で +1pp 効果**は捨てない判断もあり得るが、seat bias の悪化を許容できないため棄却

## NEXT ACTION (iter5)

1. **multi-source 値を case4 default に戻す**
2. **case7 ACCUMULATE port 本格実装** (~600行のコピペ + 配線):
   - `bot/pipeline/rulebase/case7/baseline/missions/stay.py` を `case9/baseline/missions/stay.py` にコピー
   - `case7/baseline/core/config.py` の `STAY_*` + `ACCUMULATE_*` 定数 (~65行) を `case9/baseline/core/config.py` に追加
   - `case7/baseline/strategy.py` の ACCUMULATE 配線部分を case9 に取り込む
   - `case7/baseline/strategy_helpers.py` の差分も移植
   - `ACCUMULATE_ENABLED` フラグで case9 ≡ case9-iter2 に戻せる構造を保持
3. 200戦評価で iter2 比 +2pp なら採択
