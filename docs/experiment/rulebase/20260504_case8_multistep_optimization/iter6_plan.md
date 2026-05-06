# Rulebase/case10 — iter2: t14 Trap Double Guard

> 作成日: 2026-05-05
> 関連:
> - [`iter1_plan.md`](./iter1_plan.md) / [`iter1_result.md`](./iter1_result.md) — step guard 単独で 53.0%、しきい値 -2pp 未達
> - replay 分析 (case8/9 で実証): t14 罠は 60 ships 一斉発射が原因
> スコープ: case10 上で `ACCUMULATE_KNEE_SHIPS = 60 → 40` を追加適用、step guard と二重防御

## 仮説 (Hypothesis)

iter1 の step guard (`ACCUMULATE_MIN_LAUNCH_STEP=30`) で t14 罠を消したが、`step≥30` 以降に同じ「60 ships 一斉発射」パターンが **t30+ 罠** として再発している可能性。`ACCUMULATE_KNEE_SHIPS` を 60 → 40 に下げれば、1 回あたりの発射量が減り、敵反撃で全滅するリスクが構造的に低下。step guard と二重防御で **vs v4 ≥55%** 達成を狙う。

**Mechanism**:
- step=30+ で accumulate が発動するが、KNEE=60 だと相変わらず大艦隊。敵反撃 fleet と相殺で 60→10 級の喪失パターンが残存
- KNEE=40 にすると毎回 40 ships 単位で送る → 1 回喪失しても 20 残せる、複数回打てるため fleet 再生も早い
- iter1 結果と比較すれば「step guard だけで足りるか、KNEE 削減が補完するか」が判別

## 既存コードの現状

- iter1 で実装済: `ACCUMULATE_MIN_LAUNCH_STEP=30` ガード、step guard 単独で **53.0%** (100戦)
- iter1 Stage A sweep で step=30 と step=100 が同率 53.3% → step を伸ばしても効果サチる
- KNEE=60 は case7 デフォルト、`fleet_speed` が ships 数の log で増えるため「速度を上げるための knee」として 60 が選ばれた経緯
- `_accumulate_target_threshold` 内で `max(need + SAFETY_SHIPS=4, KNEE_SHIPS=60)` → KNEE=40 にすると need < 40 の target で `threshold=40` に下がる

## スコープ (Scope)

### 変更ファイル (1 ファイルのみ)

```
bot/pipeline/rulebase/case10/baseline/core/config.py   # ACCUMULATE_KNEE_SHIPS: 60 → 40
```

その他は iter1 から不変 (step guard も維持)。

### config 変更

```python
ACCUMULATE_KNEE_SHIPS: int = 40   # iter1: 60 → iter2: 40
ACCUMULATE_MIN_LAUNCH_STEP: int = 30   # iter1 で確定、変更なし
```

## 実装ステップ

1. **`config.py:ACCUMULATE_KNEE_SHIPS = 40`** に書き換え (1 行)
2. **既存テストの再 pass 確認** — case7 から複製した `test_accumulate.py` で `KNEE_SHIPS` を直接 assert している箇所があれば monkeypatch で個別 override
3. lint / format / mypy / pytest 緑

## 検証方法

### ローカル

```bash
uv run --directory bot pytest tests/pipeline/rulebase/case10 -m "not slow" -x --no-header -q
```

### 性能評価 (30戦 + replay 分析、user 指定)

```bash
# 30戦 (replay 付き) vs v4
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v10 \
    --mode 1v1 -n 30 --seed 82000 --parallel 4
```

- **対戦相手**: `baseline_v4`
- **エピソード数**: 30 戦 (user 指定、smoke 規模)
- **主要メトリクス**: 合算勝率 (vs v4)
- **判定**: ≥55% なら採用候補 (Stage B 100戦で確認可)、50-55% なら境界線 (iter1 と差なし)、<50% なら KNEE 削減は逆効果
- **副次評価 (ログ分析)**: replay 5-10 件を `replay_to_markdown.py` で markdown 化、`step≥30` 帯で self ship_loss_burst が iter1 より減ったかを確認 (ship_loss_burst 件数 + 規模を比較)

## リスクと早期撤退条件

- **KNEE=40 で accumulate の旨味が消える**: 元々 `fleet_speed` の knee として KNEE=60 が選ばれていたので、40 では速度効率が落ちる。中盤の「40 ships で長距離発射」が遅くて到達前に敵に取られる可能性
- **step guard と冗長**: iter1 の step guard 単独で罠 trigger 70% → 0% に下げているはず。iter2 の KNEE 削減はその上澄み (~+2pp) を期待しているが、cap される可能性

## 期待される結果のシナリオ

| case10 iter2 vs v4 (n=30) | 解釈 |
|---|---|
| ≥55% | KNEE 削減が補完、Stage B 100戦に進める |
| 50-55% | iter1 と差なし、step guard 単独で十分。撤退候補 |
| <50% | KNEE 削減は逆効果 (accumulate の旨味喪失)、iter1 設定に戻す |
