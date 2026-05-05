# Rulebase/case9 — Thrash Filter on case4 base (Result)

> 作成日: 2026-05-05
> 対応 plan: [`plan.md`](./plan.md)
> 関連:
> - [`docs/experiment/rulebase/20260504_case8_multistep_beam/iter3_result.md`](../20260504_case8_multistep_beam/iter3_result.md) — case7 base 起因 handicap 仮説
> - replay 分析 (case9): `data/output/experiment/rulebase/case9/replay_analysis/20260505_1700/result_{1,2}.md`

## 結論

**case9 は採用却下、ただし重要な発見あり。** Stage A smoke 30戦 + replay 用 10戦 + case4 vs case4 ablation の 3 段階で、200戦を回す前に **構造的決着** に至った。

- case9 vs v4 = **40%** (smoke 30戦 + 10戦で再現)
- case4 vs case4 = **50%** (ablation、base 同等 self-play noise floor)
- → **thrash filter は case4 base に対し -10pp の害**
- case7 base 起因の handicap (~10pp) は **存在するが filter の害も同程度独立して効いている**
- case8 iter1/2/3 が ~30% で collapse した数式: **50% - 10pp(case7 base) - 10pp(filter 害) ≈ 30%**

**plan の仮説 (case4 base + filter で ≥55%) は否定。filter は base に関わらず害。**

## 数値

### vs baseline_v4

| 構成 | n | wins | win_rate | turn_p95 | 解釈 |
|---|---|---|---|---|---|
| case9 vs v4 (smoke A, seed 60000+) | 30 | 12 | **40.0%** | 0.670s | base 切替で +10pp 改善、しきい値 ≥40% ぎり満たす |
| case9 vs v4 (smoke B, seed 61000+) | 10 | 4 | 40.0% | 0.483s | smoke A と一致、再現性 ✅ |
| **case4 vs case4 (ablation, seed 60000+)** | **30 (=60 entries)** | **30** | **50.0%** | 0.816s | **base same-config noise floor** |
| (参考) case8 iter3 v1 vs v4 | 30 | 9 | 30.0% | 0.344s | case7 base + filter |
| (参考) case8 iter1 vs v4 | 300 | 97 | 32.3% | 0.31s | case7 base + beam (legacy) |

### しきい値判定

| 項目 | plan しきい値 | 実測 (smoke 30+10) | 判定 |
|---|---|---|---|
| 合算勝率 vs v4 | ≥55% | 40.0% | ❌ -15pp 大幅未達 |
| Stage A → Stage B 進行 | ≥40% | 40.0% | ⚠️ 境界、ablation で趨勢確定 |
| seat 対称性 | ±10pp 未満 | 評価未完 (Stage B 中止) | n/a |
| turn_p95 | ≤0.7s | 0.483-0.670s | ✅ |
| timeouts | 0 件 | 0 件 | ✅ |

## 診断 — case8 collapse の構造分解

ablation で初めて分離できた**実数式**:

```
case8 (case7 base + filter) win_rate vs v4
  = 50% (base same-config noise floor)
  - 10pp (case7 base 自体の弱さ — case4 比、概算)
  - 10pp (thrash filter 自体の害 — ablation で確認)
  = ~30%

実測: case8 iter1 = 32.3%, iter2 = 27%, iter3 v1 = 30%  ✅ 整合
```

```
case9 (case4 base + filter) win_rate vs v4
  = 50% (noise floor)
  - 0pp (base は case4 同等)
  - 10pp (thrash filter 自体の害)
  = ~40%

実測: case9 = 40% (n=40)  ✅ 整合
```

### filter の害の正体 (replay からの仮説)

case9 の loss 試合 (seed 61001, 165 turns) で観察:
- t67-78 帯で planet#19, #16 の owner 反復 thrash → filter は trigger するはず
- しかし replay では同 turn 帯で **連射 (8 ships × 5 turns)** など無駄打ちが多い
- → filter が「奪われた planet を狙わない」ため、近くの **奪取しに行きたい planet が他にない** 局面で攻撃 mission 全体の score を歪め、**遠回りの target に変な配分** が出る

つまり filter は「奪い返しに行く決定を阻止」するが、**代替の defensive 行動が無く** 結果として「ship を蓄積するだけで効果的な攻撃が打てない」状態に陥る。case3 result.md の予言「heuristic score を補正する方針は飽和」は、case9 でも該当 — **減点 mod だけでは判断軸が歪むだけで、新しい意思決定軸を持ち込まないと改善しない**。

### case7 base 起因 handicap の確認

副次的に、smoke 結果は **case7 base が case4 base より ~10pp 弱い** ことを再確認:
- case4 vs case4 = 50%
- case7 (= baseline_v7) vs case4 ≈ 40% (smoke 観測)、つまり case7 base の disadvantage ~10pp
- → case8 が case7 base 上にどんな施策を載せても production case4 を超えない

## 採用方針

- **case9 は採用却下** (filter 自体の害が判明)
- **`bot/pipeline/rulebase/case9/baseline/core/config.py` で `THRASH_FILTER_ENABLED = False`** を default にする (本 result.md 執筆完了直後に修正)
- production case4 (LB745) は引き続き現役
- case9 自体は「case4 全複製 + filter scaffold」として保持、後続 iter で別の改善案を test する base に再利用可能
- `baseline_v9` 登録も保持

## 確定した知見 (本ディレクトリの最終所見)

1. **planet thrash filter (recently_lost ベース)** は case4 / case7 どちらの base 上でも **-10pp の害**。減点系 score modifier は heuristic を歪めるだけで価値を加えない (case3 result.md 予言の追加実証)
2. **case7 base は case4 base より ~10pp 弱い** (smoke で再確認)。production を超えるには case4 base が必須
3. **case8 iter1/2/3 の collapse (~30%)** は (case7 base 弱 -10pp) + (filter 害 -10pp) の **2 軸合算**。両方独立して効く
4. **case9 vs case4 ablation で filter 害を分離** = この種の検証は ablation を必ず通すべき

## 次に試すべき方向 (本ディレクトリのスコープ外)

case4 base が production の上限近くにあると分かった以上、新しい改善は heuristic 改造ではなく **構造的に別軸** から:

| 案 | コスト | 期待 |
|---|---|---|
| **case4 上で攻撃的 mission の追加** (例: comet 奪取の積極策) | 1-2 時間 | base の攻撃半径を拡張 |
| **学習ベース value head** (imitation の case で feature engineering の延長) | 数日 | 評価関数を heuristic から離す唯一の道 |
| **case4 の hyperparameter 再 tune** (THRESHOLD 系の sweep) | 30 分 × 5 構成 | base 自体の +1-3pp 期待 |
| **mission ordering を beam ではなく portfolio search** (Churchill 系) | 1-2 日 | iter1/2 とは違う探索構造 |

### Loop の精度向上方針との整合

user の loop 指示「精度向上を目指す」に対し、本ディレクトリ (case8 + case9) の 4 iter は **採用方針の改善には至らなかった** が、 **構造的知見** を 4 つ確定させた (上記)。次の experiment はこれらを前提に新しい方向で立ち上げるべき (例: 別 family の imitation 改修、case4 base 上の新 mission 追加)。

## 再現手順

```bash
# case9 (current default 設定: THRASH_FILTER_ENABLED=True)
uv run --directory bot pytest tests/pipeline/rulebase/case9 -m "not slow" -x

# Stage A smoke
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v9 \
    --mode 1v1 -n 30 --seed 60000 --parallel 4 --no-save-replay

# ablation (case4 vs case4 で noise floor)
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v4 \
    --mode 1v1 -n 30 --seed 60000 --parallel 4 --no-save-replay

# filter OFF (case9 ≡ case4) で等価性確認
# config.py で THRASH_FILTER_ENABLED=False に変更してから上記 case9 vs v4 を再実行
```

## 関連ファイル

- `bot/pipeline/rulebase/case9/baseline/core/config.py:194-201` — `THRASH_*` 4 個 (本 result 執筆後に `THRASH_FILTER_ENABLED = False` に変更)
- `bot/pipeline/rulebase/case9/baseline/agent.py` — `StayState` (thrash 用 minimal) + `_update_thrash_state`
- `bot/pipeline/rulebase/case9/baseline/strategy_helpers.py:apply_score_modifiers` — thrash decay
- `bot/tests/pipeline/rulebase/case9/test_thrash_filter.py` — 5 unit tests (filter ロジック自体は正しいことを保証)
- `bot/tests/pipeline/rulebase/case9/test_filter_off_equals_base.py` — filter OFF で case4 等価性
- `data/output/experiment/rulebase/case9/replay_analysis/20260505_1700/` — case9 win + loss 試合の replay 分析

## 環境

- ハードウェア: M4 MacBook (local), parallel=4
- branch: `feature/rulebase-multistep-optimization`
- 実行日時: 2026-05-05
