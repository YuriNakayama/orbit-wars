# iter5 — Loop Resume State (Phase 2 DONE → Phase 3)

> 作成日: 2026-05-05
> Status: **Phase 2 完了** (stay.py 移植 + WorldModel 拡張)、Phase 3 (strategy.py 配線) を次周回で実施

## Phase 2 で完了したこと (この周回)

- `bot/pipeline/rulebase/case7/baseline/missions/stay.py` (488 行) を `case9/baseline/missions/stay.py` にコピー
- import パスは相対 import のため case9 でそのまま動作
- case9 WorldModel に `travel_time_cache` 属性 + `cached_travel_time()` メソッドを追加 (case7 から移植)
- ruff/mypy/snapshot test 3/3 全 green = iter2 と動作完全一致

## Phase 3 で次にやること (次周回)

`bot/pipeline/rulebase/case9/baseline/strategy.py` に stay.py の配線を追加:

具体的手順:
1. case7 の `strategy.py` を読み込み、stay 関連の差分 (約 36 行) を特定
2. `from .missions.stay import build_accumulate, build_stay_holds` を追加
3. `SINGLE_SOURCE_MISSION_KINDS` に `"accumulate_fire"` を追加
4. `plan_moves` 内に stay/accumulate の build と merge 処理を追加
5. `_process_single_source_mission` に accumulate_fire の分岐を追加 (case7 で 74 行目あたり)
6. case7 の `strategy_helpers.py` 差分も確認 (もしあれば移植)
7. snapshot test の **action 系列が変わるので snapshot 更新**:
   - `uv run python -m pipeline.rulebase.case9.evaluation.snapshot_update` 実行
   - 更新された snapshot を確認 → commit
8. その他の case9 単体テスト 79 件全てが pass する状態を維持

**ENABLED フラグの扱い**:
- `STAY_ENABLED=False`, `STAY_BURST_ENABLED=False`, `ACCUMULATE_ENABLED=False` のままにする (Phase 3 完了時点では動作変化なしで snapshot test 通る想定)
- Phase 4 でフラグを True に切り替えて評価

## Phase 4 (Phase 3 の次周回)

- フラグ True に変更:
  - `STAY_ENABLED=True`, `STAY_BURST_ENABLED=True`, `ACCUMULATE_ENABLED=True`
- 200戦評価: `compare_v4.py -n 100 -p 4 --seed 6000`
- ETA ~100 分

## Phase 5 (Phase 4 の次周回)

- iter5_result.md + iter5_analysis.md (replay 2-4 試合)
- 採否判定: iter2 (49.5%) 比 +2pp で採択、+5pp なら成功事例

## 過去 iter の学び

- iter1 (cooldown 抑止): 46.0%, 雪崩崩壊
- **iter2 (bypass=8 + 値短縮)**: **49.5% (best)**, 雪崩は解消
- iter3 (bypass=10 緩和): 47.8%/180中断, bypass 緩和は逆効果
- iter4 (multi-source 拡張): 47.0%, seat bias 14pp で seat=1 崩壊
- **iter5 (ACCUMULATE port)**: 余剰 ship 流用の本命、Phase 1-2 完了
