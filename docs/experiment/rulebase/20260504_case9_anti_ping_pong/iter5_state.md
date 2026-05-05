# iter5 — Loop Resume State (Phase 3 DONE → Phase 4)

> 作成日: 2026-05-05
> Status: **Phase 3 完了** (strategy.py 配線)、Phase 4 (フラグ True + 200戦評価) を次周回で実施

## Phase 3 で完了したこと (この周回)

- `strategy.py` に ACCUMULATE 配線を追加:
  - `from .core.config import ACCUMULATE_ENABLED`
  - `from .missions.stay import build_accumulate`
  - `SINGLE_SOURCE_MISSION_KINDS` に `accumulate_fire` 追加
  - `_process_single_source_mission` に accumulate_fire 分岐追加
  - `plan_moves` 内に build_accumulate 呼び出し + accumulate_holds で source_attack_left 削減 + missions に accumulate_fire 追加
- **戻り値の型は維持** (case7 のような tuple 返しではなく、案件 9 のシンプル構造を保つ)
- **STAY_BURST/DEFENSE は配線せず** (build_stay_holds は import しない、ACCUMULATE 単独テストに絞る)
- ruff/mypy/snapshot test 3/3 全 green
- ACCUMULATE_ENABLED=False のため iter2 と動作完全一致 (確認済み)

## Phase 4 で次にやること (次周回)

1. `bot/pipeline/rulebase/case9/baseline/core/config.py` で `ACCUMULATE_ENABLED: bool = False → True` に変更 (1 行)
2. snapshot test は **action 系列が変わるので失敗予定** → snapshot 更新が必要:
   - `cd bot && uv run python -m pipeline.rulebase.case9.evaluation.snapshot_update update`
   - 更新後の snapshot を git に含める
3. その他のテスト 78 件は pass する見込み (ACCUMULATE が攻め優先で fail する可能性は低い)
4. 200戦評価をバックグラウンドで起動: `nohup uv run python -m pipeline.rulebase.case9.evaluation.compare_v4 -n 100 --seed 6000 -p 4 > /tmp/compare_v4_iter5.log 2>&1 &`
5. ETA ~100 分、次周回でガード発動

## Phase 5 (Phase 4 + 評価完了の次周回)

- iter5_result.md 作成 (採否判定: iter2 比 +2pp)
- `/experiment-analysis` で iter5_analysis.md (replay 比較)
- 1 commit に統合 + push
- 採択なら memory に成功事例記録、`/loop` 完了条件達成判定

## Phase 5 の判断 tree

- **+5pp (≥54.5%)** : 採択 + completion 条件達成 → loop 停止 (CronDelete 11c931e9)
- **+2pp (≥51.5%)** : 採択 (best 更新)、iter6 で更にチューニング (ACCUMULATE_KNEE_SHIPS, MIN_TARGET_TURNS)
- **<+2pp** : 棄却、ACCUMULATE_ENABLED=False に戻して iter6 で別軸 (capture 強化、speed 最適化)

## 過去 iter サマリ

- iter1 (cooldown 抑止): 46.0%
- **iter2 (bypass=8 + 値短縮)**: **49.5% (best)**
- iter3 (bypass=10): 47.8%/180中断
- iter4 (multi-source): 47.0%, seat bias 14pp
- **iter5 (ACCUMULATE port)**: 進行中 — Phase 1-3 完了、Phase 4 で評価予定
