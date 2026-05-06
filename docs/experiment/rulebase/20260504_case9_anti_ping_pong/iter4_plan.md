# rulebase/case9 — anti_ping_pong (iter4 plan)

> 作成日: 2026-05-05
> 関連: `iter1-3_*.md`, replay 比較分析 `data/output/experiment/rulebase/case9/replay_analysis/20260505_0700/`
> スコープ: multi-source swarm 拡張で「t=100 大型 launch」促進 (ACCUMULATE 全体 port は iter5 以降に分離)

## 仮説 (Hypothesis)

直近 replay 比較で **win 試合 (seed 3052) は t=100 で 100+ ships の大型 launch を 1 撃成功させており、これが分岐点**であることを発見。loss 試合は同 turn 帯で大型 launch なし → 雪崩崩壊。
**`MULTI_SOURCE_TOP_K=5→8` + `THREE_SOURCE_PLAN_PENALTY=0.75→0.85`** に緩和することで multi-source swarm の発火条件を増やし、中盤での大型 launch を促進すれば、**loss シナリオの一部を win に転換** できる。

このアプローチは案件の本命 (ACCUMULATE port) より小規模だが、**同じ方向の介入** (余剰 ship を大型 launch に転用) を 2 行で試せる。効果が確認できれば iter5 で本格 ACCUMULATE port、効果なしなら 別軸 (capture 強化など) に切り替える早期判断材料になる。

## スコープ (Scope)

**変更ファイル**: `bot/pipeline/rulebase/case9/baseline/core/config.py` のみ
- `MULTI_SOURCE_TOP_K: 5 → 8` (swarm 候補 src を増やす)
- `THREE_SOURCE_PLAN_PENALTY: 0.75 → 0.85` (3-source swarm の score penalty を緩めて発火しやすく)
- `LOW_PLANET_BYPASS_THRESHOLD` は **8 のまま維持** (iter2 の知見)

**変更しないファイル**: それ以外すべて。ACCUMULATE port、agent 速度最適化、cooldown 値はこの iter のスコープ外。

## 実装ステップ (Implementation outline)

1. config.py の 2 定数を更新
2. `dev/lint` で case9 のみチェック
3. `pytest tests/pipeline/rulebase/case9 -x` (snapshot test 含む)
4. 200戦評価: `compare_v4.py -n 100 -p 4 --seed 5000`
5. iter4_result.md 作成、採否判定

## 検証方法 (Validation method)

- 評価対戦相手: baseline_v4 (case4)
- エピソード: 200戦 (seed 5000-5199)
- 主要メトリクス: vs v4 勝率
- しきい値:
  - **iter2 比 +2pp 以上 (= 51.5%)** で採択 → iter5 で ACCUMULATE port を積み増し
  - +2pp 未満 → 棄却して iter5 で別軸 (ACCUMULATE port または capture 強化)
- 補助メトリクス: 試合の `enemy_planet_attack 100+ ships` 発生回数 (replay 経由で iter5 以降)

## 想定リスク

- **swarm 過多**: TOP_K=8 + penalty 緩和で並列 swarm が多発し、reinforce/defense が手薄になる可能性
- **case4 系列の調整値破壊**: case4 (production) の 70.3% vs v3 を維持している tuned 値を弄るリスク。bypass=8 + cooldown 短縮は維持しているので相互作用は要観察

## 引き継ぎ (NEXT for iter5)

- iter4 採択 (≥51.5%) → ACCUMULATE port (case7 stay.py 系を case9 に移植)、bypass=8 + multi-source 強化を維持
- iter4 棄却 (<51.5%) → multi-source 設定を元に戻し、capture mission の score weight 強化 or ACCUMULATE port 単独を主役に
