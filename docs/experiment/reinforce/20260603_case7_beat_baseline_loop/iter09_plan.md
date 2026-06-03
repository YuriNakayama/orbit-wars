# case7 「ルールベースに勝つ」ループ — iter09 PLAN

時刻: 2026-06-03 10:14 (cron tick 11)

## ここまで (11 tick, ~10 variant)
全て vs baseline_v1 = 0/10。production shaping でも score gap 51:13834 と大差。
RL machinery は健全 (kl>0, vs self reward+, beats random 10/10)。parity 81件 pass。
web search: Generals.io 論文が同レシピで H100×36h を要した = **scale が 1000倍不足**が根因。

## 残る打ち手の評価
| 案 | 小規模? | コスト | 期待値 |
|---|---|---|---|
| memory features 修正 (empty history → 実 launch) | ○ | 無料 | 論文の加速要因。ただし JIT scan 内 + parity リスク大、単独で 0/10→勝利は望み薄 (論文も memory+scale 両方必要) |
| GPU 大規模 (RunPod) | × (中規模) | ~$1.5+/run | 研究が示す本筋。ただし memory: case6 は GPU でも本物 v1 に 0/10。push 必要 |
| 別 family (rulebase/case8) | ○ | 無料 | 既に v1 互角〜上。「v1 に勝つ」goal なら確実 |

## iter09 方針
小規模・無料・低リスクを優先する loop 趣旨に従い、以下を順に:

1. **memory features 修正を実装・検証** (iter09a):
   学習 rollout の `update_history_jax` に empty でなく **実 launch** (target_slot!=NO_OP の
   from_planet_id / ships / valid) を渡す。parity への影響を smoke で確認。
   → 完走 → 10戦 vs v1。動けば論文の memory features 効果を小規模で実証。

2. memory features でも 0/10 なら、**scale が唯一の道**と確定 → GPU 判断を仰ぐ:
   - 要 push (branch を origin へ) + RunPod $1.5+/run + 結果は不確実 (memory)。
   - これは重い一歩 (remote push + 課金) なのでユーザー greenlight を得てから。

## 留意
- これ以上「同レシピの小規模 RL を回す」だけは 0/10 確定で無駄打ち。新規性のある
  変更 (memory features) か scale up のどちらかに絞る。
- commit c68cbecc は local のみ (未 push)。
