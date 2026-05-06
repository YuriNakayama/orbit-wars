# Abandoned case10 — capture_kamikaze 探索の撤退まとめ

> 作成日: 2026-05-06
> このブランチ (`feature/rulebase-planet-ping-pong`) では case9 を最終 case として残し、case10 は撤退する。本ファイルは case10 探索の知見を case9 docs 配下に保管する目的のもの。

## 背景

case9 anti-ping-pong 9 iter 全棄却を踏まえ、別軸として case4 base 上の **score weight 調整 + 攻撃寄り設定** を case10 で試行。

## case10 試行サマリ (2 iter で撤退)

### iter1 — 5 定数で capture 強化 + kamikaze 多用

| 定数 | case4 default | iter1 |
|---|---|---|
| `STATIC_NEUTRAL_VALUE_MULT` | 1.4 | 1.6 |
| `EARLY_NEUTRAL_VALUE_MULT` | 1.2 | 1.4 |
| `SNIPE_VALUE_MULT` | 1.12 | 1.30 |
| `HARASS_MIN_SRC_RESERVE` | 10 | 6 |
| `HARASS_PRODUCTION_STEAL_TURNS` | 5 | 8 |

**結果**: 200戦 **45.5%** (棄却、case4 -4pp)

### iter2 — 逆方向 (defense 寄り)

| 定数 | iter1 | iter2 |
|---|---|---|
| `STATIC_NEUTRAL_VALUE_MULT` | 1.6 | **1.2** (default 1.4 より低く) |
| `HARASS_MIN_SRC_RESERVE` | 6 | **14** (default 10 より高く) |
| 残り 3 定数 | iter1 値 | default 復元 |

**結果**: 200戦 **48.5%** (棄却、iter1 比 +3pp も case4 default 未達)

## 撤退判断

両方向 (capture 強化 / 弱化) とも case4 default を超えず、**case4 default は両側ともに既に tuned** と確定。memory `project_heuristic_search_saturation` の「heuristic 系探索は 53% で飽和」を再確認。

## 教訓 (新規)

1. **5 定数同時変更で全方向に反転しても結果不変** = state space が変わらない、heuristic 系の本質的限界
2. **score weight 調整は score の絶対値を動かすだけ**、mission 順序は変わらず実質的な action 変化は起きない
3. **次の有効手は学習評価関数 / 別 family** (memory 教訓と整合)

## 撤退で削除されたもの

- `bot/pipeline/rulebase/case10/` (case4 フルコピー + 5 定数変更版)
- `bot/tests/pipeline/rulebase/case10/`
- `AGENT_REGISTRY` の `baseline_v10`
- `bot/pyproject.toml` の case10 lint 例外
- `docs/experiment/rulebase/20260506_case10_capture_kamikaze/` (本ファイルに統合)

## このブランチの最終 case = case9

`docs/experiment/rulebase/20260504_case9_anti_ping_pong/` の iter1-9 を案件の本流とし、case9 をブランチの代表 case として PR 作成可能。採用された改善:
- iter6 `plan_shot` 1-turn memoize (30% 高速化、ablation 採用)
- iter2 設計 (`bypass=8` + cooldown 1/2 + REINFORCE_MIN_DEFICIT=1) は best 設計だが +5pp 未達のため case4 production を置き換えない
