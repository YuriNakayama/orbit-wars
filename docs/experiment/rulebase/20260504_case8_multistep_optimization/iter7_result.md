# Rulebase/case10 — iter3 Result: Phase 5 Thrash Suppression

> 作成日: 2026-05-05
> 対応 plan: [`iter3_plan.md`](./iter3_plan.md)
> 関連:
> - [`iter1_result.md`](./iter1_result.md) — step guard 単独で 53.0% (n=100)
> - [`iter2_result.md`](./iter2_result.md) — KNEE=40 は -3pp 逆効果
> - replay 分析: `data/output/experiment/rulebase/case10/replay_analysis/20260505_iter3/result_{1,2}.md`

## 結論

**仮説は機能的には支持、勝率改善はゼロ。iter3 は採用却下。** 30 戦 vs `baseline_v4` で **53.3%** (16/30)、iter1 (Stage A 30戦 53.3%) と完全同率。replay 分析で **planet_loss 件数は iter2 比 -36% 削減 (42 → 27)** されており、thrash filter は意図通り機能。だが **勝率には貢献していない**。

case8 iter3 v1 / case9 で観察した「filter は thrash 件数を減らすが勝率に効かない」パターンが case10 でも再現。**減点系 score modifier の限界が再確認**された。

## 数値

### iter3 30戦結果

| | wins (30戦, seed 83000+) | win_rate | turn_p95 |
|---|---|---|---|
| baseline_v10 (iter3) | 16 | **53.3%** | 0.365s |
| baseline_v4 | 14 | 46.7% | 0.789s |

### iter1/2/3 比較 (同 30戦規模)

| iter | win_rate | turn_p95 | replay 上の挙動 |
|---|---|---|---|
| iter1 (step guard) | 53.3% | 0.34s | t14 罠 0、Phase 5 thrash あり |
| iter2 (+KNEE=40) | 50.0% | 0.69s | t14 罠 0、捕獲効率低下 |
| **iter3 (+thrash repeat filter)** | **53.3%** | 0.37s | **t14 罠 0、planet_loss -36%、勝率変わらず** |

### しきい値判定

| 項目 | しきい値 | 実測 | 判定 |
|---|---|---|---|
| 合算勝率 vs v4 | ≥55% | 53.3% | ❌ -1.7pp 未達 |
| iter1 比改善 | +2pp | 0pp | ❌ 並列 |
| t14 罠抑制 | replay で 0 件 | 0 件 ✅ |
| Phase 5 thrash 抑制 | planet_loss 削減 | 42→27 (-36%) ✅ |

## 診断 — 機能 vs 勝率の乖離

### Filter は意図通り機能した

iter3 long match (seed 83001) replay 集計:
- self planet_loss: **27** (iter2 long match の 42 から **-36%**)
- 最頻 planet (#29 / #23 / #17 を 6 回 thrash) は iter2 (#2 を 12 回 thrash) より **半分**
- 「奪われた直後の取り返し」mission が `THRASH_REPEAT_LIMIT=2` で抑制されている

### しかし勝率は不変

iter3 long match の最終: planets 3 vs 28、ships 1350 vs 35387、production 3 vs 84 → **完敗**。iter2 long の 3 vs 29 / 706 vs 33004 / 3 vs 101 と **本質的に同じ敗戦パターン**。

**主因 (replay からの仮説)**:
1. **Filter で取り返しを諦めた planet は敵のものになる**: thrash 連鎖を抑えても、敵が安定して保持する planet は増える。production 比 3:84 は変わらない
2. **代替 target が無い**: filter で抑制された capture/snipe/swarm の代わりに別 target に振り向ける動きが弱い。中盤の **planet 拡張ペースが落ちる** = production 増分が減る
3. **本質的問題**: case7 base の `accumulate_fire` 系が中盤以降も「60 ships を遠距離に投げる」設計、敵反撃で大半失う構造は filter で解消されない

つまり **replay 分析で見えた Phase 5 thrash は症状、根本原因は Phase 3 (t30+ 罠) と中盤の accumulate 設計**。filter で症状を抑えただけでは底上げにならない。

### case8/9/10 累計の filter 系試行

| 試行 | filter 種別 | win_rate | filter 機能 |
|---|---|---|---|
| case8 iter3 v0 | recently_lost + 暴走 commits (bug) | 26% | 機能不全 |
| case8 iter3 v1 | recently_lost only | 30% | thrash -76% だが勝率薄 |
| case9 | recently_lost (case4 base) | 40% | filter 害 -10pp |
| **case10 iter3** | **mission_commits (種別絞り、bug 修正版)** | **53.3%** | **planet_loss -36% だが勝率変わらず** |

→ **減点系 score modifier は構造的に勝率改善に繋がらない** が **改めて経験的に裏付け**。`project_thrash_filter_harm.md` memory の知見と一致。

## 採用方針

- **iter3 は採用却下**
- `bot/pipeline/rulebase/case10/baseline/core/config.py` の `THRASH_REPEAT_FILTER_ENABLED=True` を **False に戻す** (本 result 執筆完了直後に修正)
- case10 確定構成: **iter1 設定 (step guard=30, KNEE=60, no repeat filter) で 53.0% (n=100)**

## 確定した知見

1. **Phase 5 thrash は症状、根本原因ではない**: filter で抑えても勝率改善せず、敵が拡張するだけ
2. **減点系 score modifier の限界が再確認**: 6 連敗 (OM v1/v2, lookahead, beam reorder, thrash by-loss, thrash by-commits) → `project_thrash_filter_harm.md` の知見を強化
3. **case10 の真の弱点は Phase 3 の t30+ 罠 + 中盤 accumulate 設計**: filter で症状を抑えるのではなく、accumulate logic 自体を見直す必要

## 次の方向 (本ディレクトリのスコープ外)

heuristic 系の改修はもうない:

| 案 | 期待 | コスト | 推奨度 |
|---|---|---|---|
| accumulate logic の根本改修 (KNEE_SHIPS を target need ベースに動的化) | 中盤発射量を「必要分のみ」に絞る | 1-2時間 | ★★★ |
| case10 200戦で再評価 (iter1 53.0% の境界線確定) | seed variance 縮小、≥55% 達成判定 | 30分 | ★★ |
| 学習評価関数 (case4 base 上) | heuristic 飽和を脱出 | 数日 | ★ (本ディレクトリ外) |

## 関連ファイル

- `bot/pipeline/rulebase/case10/baseline/core/config.py:THRASH_REPEAT_*` — iter3 で追加、本 result で `THRASH_REPEAT_FILTER_ENABLED=False` に変更
- `bot/pipeline/rulebase/case10/baseline/agent.py:record_mission_commit` — mission_commits 記録関数
- `bot/pipeline/rulebase/case10/baseline/strategy.py:_process_*_mission` — capture/snipe/swarm commit 時の `record_mission_commit` 呼び出し
- `bot/pipeline/rulebase/case10/baseline/strategy_helpers.py:apply_score_modifiers` — thrash decay block
- `data/output/experiment/rulebase/case10/replay_analysis/20260505_iter3/` — iter3 long+fastest_loss replay
