# rulebase/case9 — anti_ping_pong (iter3 plan)

> 作成日: 2026-05-05
> 関連: `iter2_*.md`, `iter1_*.md`
> スコープ: bypass threshold 緩和の単独効果を測定 (ACCUMULATE port は iter4 に分離)

## 仮説 (Hypothesis)

iter2 analysis で「拮抗持続シナリオ C で 16-19 惑星帯の僅差負けが残存」を観測。
**`LOW_PLANET_BYPASS_THRESHOLD=8 → 10` に緩和**することで、劣勢シグナルを早めにキャッチして cooldown を bypass し、シナリオ C の僅差負けの一部を引き分けまたは勝利に転換できる。

iter3 では bypass 緩和単独の効果を測定し、効果が +2pp 以上なら採択 (cumulative で iter1+2+3 の +5pp 達成可能性)。それ以下なら iter4 で ACCUMULATE port に進む。

## スコープ (Scope)

**変更ファイル** (1 行のみ):
- `bot/pipeline/rulebase/case9/baseline/core/config.py`:
  - `LOW_PLANET_BYPASS_THRESHOLD: int = 8 → 10`

**変更しないファイル**: それ以外すべて。ACCUMULATE port、cooldown 値、agent 速度最適化はこの iter のスコープ外。

## 実装ステップ (Implementation outline)

1. config.py の `LOW_PLANET_BYPASS_THRESHOLD` を 8 → 10 に変更
2. `pytest tests/pipeline/rulebase/case9 -x` で 79/79 pass 確認
3. **200戦評価**: `compare_v4.py -n 100 -p 4 --seed 4000` (新 seed range で iter2 と独立評価)
4. iter3_result.md を書く
5. 採否判定:
   - +2pp 以上 (iter2 49.5% → 51.5% 以上) → 採択、iter4 で ACCUMULATE port に積む
   - +2pp 未満 → 棄却して iter4 で ACCUMULATE port を主役に

## 検証方法 (Validation method)

- ローカル: `dev/test-bot` (lint/type/pytest)
- 評価:
  - 対戦相手: baseline_v4 (case4)
  - エピソード: 200戦 (各 seat 100戦、seed 4000–4199)
  - 主要メトリクス: vs v4 勝率
  - しきい値: **iter2 比 +2pp で採択** (cumulative 改善方向の確認)
- リモート: 不要

## 想定リスク

- **bypass 過剰**: 10 惑星でも bypass が走ると ping-pong 抑止効果が弱まる可能性 → diagnose_ping_pong での件数確認は iter4 以降

## 引き継ぎ (NEXT for iter4)

iter3 の結果に応じて:
- **iter3 採択** (51.5%以上): bypass=10 維持 + case7 ACCUMULATE port を iter4 で実施
- **iter3 棄却** (51.5%未満): bypass を 8 に戻し、ACCUMULATE port を iter4 主役に
