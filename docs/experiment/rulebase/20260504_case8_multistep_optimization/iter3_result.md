# Rulebase/case8 — Planet Thrash Filter (iter3 result)

> 作成日: 2026-05-05
> 対応 plan: [`iter3_plan.md`](./iter3_plan.md)
> 関連:
> - [`iter1_result.md`](./iter1_result.md), [`iter2_result.md`](./iter2_result.md) — beam 系飽和
> - replay 分析 (iter2): `data/output/experiment/rulebase/case8/replay_analysis/20260505_1410_iter2/`
> - replay 分析 (iter3 v0): `data/output/experiment/rulebase/case8/replay_analysis/20260505_1530_iter3/`
> - replay 分析 (iter3 v1): `data/output/experiment/rulebase/case8/replay_analysis/20260505_1600_iter3v1/`

## 結論

**iter3 は採用却下。Stage A で打ち切り、200戦評価 (Stage B) は実施せず。**

thrash filter (recently_lost + mission_commits) を実装、smoke で **vs v4 = 26-30%** に留まり、しきい値 ≥40%(Stage A) に未達。**filter 自体は thrash 件数を 187→45 (76% 削減) と機能している** が、勝率改善には繋がらず → planet thrash は敗因の一部だが主因ではないことが判明。

iter1 (32.3%) / iter2 (27.0%) / iter3 (26-30%) すべて **同じ局所最適 (~30%)** に collapse、greedy heuristic を表層的に修正する系の方針が飽和したことが明確になった。

## 数値

### 主要メトリクス: vs baseline_v4

| 構成 | n | v4 wins | v8 wins | **v8 win_rate** | v8 turn_p95 | thrash events (long match) |
|---|---|---|---|---|---|---|
| iter1 (basis) | 300 | 203 | 97 | 32.3% | 0.31s | 50戦 collected, ~94 events |
| iter2 (true2p_sampled) | 100 | 73 | 27 | 27.0% | 0.57s | 83 |
| **iter3 v0 (recently_lost + 暴走 commits)** | 50 | 37 | 13 | **26.0%** | 0.71s | **187 (悪化)** |
| **iter3 v1 (recently_lost only)** | 30 | 21 | 9 | **30.0%** | 0.34s | **45 (改善)** |

### Stage A しきい値

`iter3_plan.md` 設計: vs v4 ≥40% かつ thrash 件数減少 → Stage B (200戦) へ進む。

- **iter3 v0**: 26% (-14pp 未達) かつ thrash 187 (3倍化) → 即停止 + bug 修正
- **iter3 v1**: 30% (-10pp 未達) かつ thrash 45 (改善) → filter 機能は確認、勝率改善せず → 撤退

## 診断

### iter3 v0 で発生した 3 つの実装バグ

#### Bug 1: `_record_mission_commits` が move 種別を区別しなかった (主因)

```python
# v0 の `agent.py:_record_mission_commits`
for move in moves:
    target = _infer_action_target(src, angle, world.planets, ships)
    if target is None or target.owner == world.player:
        continue
    _STAY_STATE.mission_commits.setdefault(target.id, []).append(step)
```

問題: capture/snipe/swarm だけでなく **harass / accumulate_fire / followup / evacuation / rear_guard** から emit された全 move を記録。
- harass は production を奪う目的で意図的に "奪われる" mission
- accumulate_fire は multi-turn で同 planet を狙い続ける
- followup/evacuation/rear_guard は防衛系の動き

結果: ほぼ全ての敵/中立 planet が `mission_commits >= 2` を window 内に超え、**filter が全 capture mission を一律 0.3 倍に減衰** → score 序列が壊れて greedy が不適切な ordering に → thrash がむしろ増加。

#### Bug 2: `_infer_action_target` の angle-cone 推論精度

`predict_enemy_reaction` から流用した angle ray + radius intersection で「最近接の交差 planet」を返すが、長距離 fleet では intended target を超えて他 planet に交差する場合がある。**Bug 1 と組み合わさって誤記録を増幅**。

#### Bug 3: 同 planet への multi-turn 連続発射の snowball

`accumulate_fire` のような mission が同 planet を毎ターン狙い続けると、`mission_commits[planet_id]` が毎ターン append。window=10 内で 10 commits 蓄積 → 永続的に thrash 認定。

### v1 修正と検証

`THRASH_REPEAT_COMMIT_LIMIT = 999` で **mission_commits 経路を無効化**、`recently_lost` 単独で判定:

- ✅ thrash 件数 187 → 45 (76% 削減) 確認 = filter は意図通り機能
- ❌ win_rate 26% → 30% (+4pp) 程度に留まる、しきい値 ≥40% 未達

→ **「奪われた planet 即奪い返し」抑制だけでは vs v4 に対し勝てない**。replay 分析で見えた他の構造要因 (序盤 t14 の ship 枯渇、case7 base 自体の弱さ) が支配的。

### iter1/2/3 の collapse パターン (~30% で停滞)

| 試行 | 介入箇所 | 結果 |
|---|---|---|
| iter1 | beam search (legacy_net_ships) | 32.3% |
| iter2 | beam + true2p_sampled + mission_score | 27.0% |
| iter3 v0 | greedy + 暴走 thrash filter | 26.0% |
| iter3 v1 | greedy + recently_lost only | 30.0% |

すべて **同じ局所最適 (~30%) に collapse**。これは `case3 result.md` の予言「heuristic score を補正する方針は飽和」を 3 連敗で経験的に裏付けた結果。

## 採用方針

- **iter3 は採用却下**
- `bot/pipeline/rulebase/case8/baseline/core/config.py` の `BEAM_ENABLED=False` (iter2 撤退時に既設定) + `THRASH_REPEAT_COMMIT_LIMIT=999` の状態で **Kaggle 提出には使えない案として維持** (`THRASH_FILTER_ENABLED=True` のままだが filter は recently_lost のみ動作、効果薄でも害は無い)
- production case4 (LB745) は引き続き現役
- iter3 で導入した owner 遷移検出 + thrash decay 基盤は iter4 で再利用可能

## iter1+iter2+iter3 を通じて確定した構造的所見

1. **case7 base 自体が v4 に対し劣勢** (smoke で v4 vs v7 = 60-40 観測) — どの上位施策を載せても base の劣化は埋められない
2. **序盤 t14 の自軍 ship 大量損失** が iter1/2/3 共通で再現 (replay 全件で観測) — beam / thrash filter のいずれも触れていない構造
3. **heuristic mission.score の補正系列** (beam, evaluator, thrash decay) は飽和 → 候補生成自体を変えるか、評価関数を learning-based に置換するしかない

## iter4 で試す価値があるもの (本実験ディレクトリでの最終提案)

優先順序:

1. **base を case4 に切り替え** (新 case9 として、case4 全複製 + thrash filter v1 を被せる) — case7 base 起因の handicap (~10pp) を切り分ける最小コスト実験。1 時間
2. **序盤 (t<20) の attack mission を抑制** — t14 ship 枯渇は序盤に攻撃 mission を出しすぎている疑い。`SAFE_OPENING_*` 系 config を絞って測定。30 分
3. **学習評価関数 (imitation の value head 流用)** — heuristic 飽和を完全に脱出する方向。数日コスト、本ディレクトリのスコープ外、新規 plan が必要

iter4 を案件として継続するか別方向 (例: 別 family の experiment) に切り替えるかは user 判断。

## 採用済み memory への影響

新規 memory 候補: 「**case8 で iter1 (beam) → iter2 (beam+true2p) → iter3 (thrash filter) を 3 連敗、いずれも vs v4 ~30% で停滞。case7 base 自体の劣化と t14 序盤 ship 枯渇が主因仮説**」を `project_case8_iter123_collapse.md` で記録すべき (user 判断)。

## 再現手順

```bash
# iter3 v1 (default)
uv run --directory bot pytest tests/pipeline/rulebase/case8 -m "not slow" -x

uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v8 \
    --mode 1v1 -n 30 --seed 53000 --parallel 4

# iter3 v0 (再現するには) — config.py で THRASH_REPEAT_COMMIT_LIMIT=2 に戻す
```

## 関連ファイル

- `bot/pipeline/rulebase/case8/baseline/core/config.py:280-290` — `THRASH_*` 4 個 (REPEAT_COMMIT_LIMIT=999 が v1 撤退設定)
- `bot/pipeline/rulebase/case8/baseline/agent.py` — `_update_thrash_state`, `_record_mission_commits` (commits 経路は無効化されているが残置)
- `bot/pipeline/rulebase/case8/baseline/strategy_helpers.py:apply_score_modifiers` — thrash decay (recently_lost 単独で動作)
- `bot/tests/pipeline/rulebase/case8/test_thrash_filter.py` — 5 unit tests (filter ロジック自体は正しいことを保証)
- `data/output/experiment/rulebase/case8/replay_analysis/20260505_15{30,30_iter3,1600_iter3v1}/` — iter2/iter3v0/iter3v1 の replay 比較

## 環境

- ハードウェア: M4 MacBook (local), parallel=4
- branch: `feature/rulebase-multistep-optimization`
- 実行日時: 2026-05-05
