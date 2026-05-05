# iter5 — Loop Resume State

> 作成日: 2026-05-05
> Status: **iter5 未開始** — multi-source 設定は iter2 同等に復元済み

## 完了したこと (iter4 締め)

- iter4 棄却 (47.0% / 200戦、seat bias 14pp が主因)
- multi-source 設定を case4 default に戻した:
  - `MULTI_SOURCE_TOP_K: 8 → 5`
  - `THREE_SOURCE_PLAN_PENALTY: 0.85 → 0.75`
- 現状の case9 = iter2 同等 (bypass=8 + cooldown 1,2 + REINFORCE_MIN_DEFICIT=1)

## iter5 でやること (大規模、複数周回かかる)

### Phase 1 (1 周回): ACCUMULATE 関連定数を case9/config.py に追加

- case7 config.py の `STAY_*` (24行) + `ACCUMULATE_*` (24行) を case9 にコピペ
- 配線はまだしないので動作変化なし (定数だけ存在する状態)
- pytest で snapshot test 通過確認

### Phase 2 (1 周回): stay.py を case9 に追加

- `bot/pipeline/rulebase/case7/baseline/missions/stay.py` (488行) を `case9/baseline/missions/stay.py` にコピー
- import パス調整 (相対 import に書き換え)
- まだ strategy.py で呼ばない、定義だけ。pytest 通過

### Phase 3 (1 周回): strategy.py + strategy_helpers.py 配線

- `bot/pipeline/rulebase/case7/baseline/strategy.py` の ACCUMULATE 関連 (collect_missions の中の build_stay_holds / build_accumulate 呼び出し) を case9/strategy.py に取り込む
- `SINGLE_SOURCE_MISSION_KINDS` に `accumulate_fire` 追加
- `strategy_helpers.py` の差分関数 (もしあれば) も移植
- `ACCUMULATE_ENABLED=True` で動作、`False` で iter2 等価
- pytest で snapshot 更新が必要 (action 系列が変わる)

### Phase 4 (1 周回): 200戦評価

- `compare_v4.py -n 100 -p 4 --seed 6000`
- ETA ~100 分

### Phase 5 (1 周回): result + analysis + commit

- 採択しきい値: iter2 比 +2pp (51.5%) で採択 → memory 候補 / +5pp 達成なら成功事例
- 棄却なら ACCUMULATE 関連定数のチューニング (iter6+)

## 過去 iter の学び (引き継ぎ用)

- iter1 (cooldown 抑止): 46.0%, 雪崩崩壊シナリオ
- **iter2 (bypass=8 + 値短縮)**: **49.5% (best)**, 雪崩は解消
- iter3 (bypass=10 緩和): 47.8%/180中断, bypass 緩和は逆効果
- iter4 (multi-source 拡張): 47.0%, seat bias 14pp で seat=1 が崩壊
- iter5 (ACCUMULATE port): 余剰 ship 流用の本命、replay 分析で支持される設計
