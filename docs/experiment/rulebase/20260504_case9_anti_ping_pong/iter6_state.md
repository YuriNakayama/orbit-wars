# iter6 — Loop Resume State

> 作成日: 2026-05-05
> Status: **iter6 未開始**、case9 は iter2 等価に復帰済み

## 直近の状態 (iter5 締め)

- iter5 ACCUMULATE port: **42.5% (200戦)、過去最低 → 棄却**
- 原因: ACCUMULATE 単独 (STAY_BURST 抜き) では中盤の攻めを hold で止めて逆効果
- 復元済み: `ACCUMULATE_ENABLED=False`、snapshot 更新、case9 = iter2 等価

## iter6 候補 (優先順)

### 候補 A: agent 速度最適化 (推薦、別 commit、ablation 必須)

- `WorldModel.__init__` の重複計算を cache:
  - `base_timeline` の 2 重計算がある可能性 (要確認)
  - `indirect_wealth` 計算の memoize
- `plan_shot` の早期 return: 明らかに不可能なケースを最初に弾く
- 評価: ablation で iter2 比 ±2pp 以内 (300戦) なら採用、勝率変化なら棄却
- 副次効果: iter6 以降の評価が高速化

### 候補 B: iter2 cooldown 値の微調整

- `PING_PONG_PAIR_COOLDOWN_TURNS: 1 → 2` (iter1 と iter2 の中間)
- 評価: 200戦で iter2 比 +2pp なら採択
- リスク: iter2 の知見と矛盾する可能性

### 候補 C: rust simulator 切替検討 (要 user 判断)

- rustup 導入 → maturin develop で 200戦 ~5 分に短縮
- 300戦評価が現実的になり、iter2 best の信頼区間を狭められる
- ユーザー権限要

### 候補 D: ACCUMULATE + STAY_BURST 同時 port (大型 iter)

- case7 の STAY_BURST 配線も加える (~200 行追加)
- 1 周回では完結せず、5+ 周回必要

## 推薦判断

**候補 A (速度最適化) を iter6 で実施**。理由:
- 過去 iter で評価コストが最大の制約
- 速度向上は採否しきい値に直接影響しない (ablation で性能維持を保証)
- iter7 以降の試行回数を増やせる

候補 B/C は iter6 採否 (速度最適化が成功) 後に検討。
