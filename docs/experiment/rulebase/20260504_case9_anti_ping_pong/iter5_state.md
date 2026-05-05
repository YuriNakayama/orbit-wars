# iter5 — Loop Resume State (Phase 1 DONE → Phase 2)

> 作成日: 2026-05-05
> Status: **Phase 1 完了**, Phase 2 (stay.py 追加) を次周回で実施

## Phase 1 で完了したこと (この周回)

- case7 の STAY_* (10 定数) と ACCUMULATE_* (9 定数) を case9/config.py に追加
- 全フラグ `STAY_ENABLED=False`, `STAY_BURST_ENABLED=False`, `ACCUMULATE_ENABLED=False` で初期化
- pytest 3/3 pass = iter2 と動作完全一致 (定数追加のみで挙動変化なし)
- ruff/mypy green

## Phase 2 で次にやること (次周回)

**`bot/pipeline/rulebase/case7/baseline/missions/stay.py` (488 行) を case9 にコピー + import パス調整**

具体的手順:
1. `cp bot/pipeline/rulebase/case7/baseline/missions/stay.py bot/pipeline/rulebase/case9/baseline/missions/stay.py`
2. case9/missions/stay.py 内の import 文を確認:
   - `from ..core.config import ...` の相対 import はそのままで OK
   - `from ..core.types import ...`, `from ..core.world_model import ...` も OK
   - **case6/case7 固有の import (例: 特殊な safety helper) があれば、case9 に同等関数があるか確認**
3. `__init__.py` (missions) で stay を export しない (Phase 3 で配線時に追加)
4. ruff check / mypy で import エラーが無いことを確認
5. pytest tests/pipeline/rulebase/case9 -x で snapshot 含む全 79 件 pass を確認 (Phase 2 では呼ばれないので動作変化なし)

## Phase 3 (Phase 2 の次周回)

`strategy.py` で stay を配線。詳細は次周回で iter5_state.md を更新。

## Phase 4-5

200戦評価 + result + analysis + commit

## 過去 iter の学び

- iter1-4 結果は iter4_result.md / iter5_state.md (旧版) 参照
- best: **iter2 49.5%** (200戦)、本命の ACCUMULATE port (iter5) で +5pp 達成を狙う
