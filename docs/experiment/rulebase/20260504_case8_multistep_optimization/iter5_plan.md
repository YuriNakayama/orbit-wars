# Rulebase/case10 — Accumulate Step Guard (t14 trap fix)

> 作成日: 2026-05-05
> 関連:
> - [`docs/experiment/rulebase/20260504_case8_multistep_beam/iter3_result.md`](../20260504_case8_multistep_beam/iter3_result.md) — case8 iter1/2/3 collapse 全敗
> - [`docs/experiment/rulebase/20260505_case9_thrash_filter_on_case4/result.md`](../20260505_case9_thrash_filter_on_case4/result.md) — filter は base 関わらず -10pp 害
> - replay 分析 (case9 vs case7): `data/output/experiment/rulebase/case9/replay_analysis/20260505_1730_v9_vs_v7_all10/` (10戦集計、t14 罠 trigger 70%)
> - memory: `project_case7_t14_trap.md`, `project_thrash_filter_harm.md`
> スコープ: 新規 `case10` を切り、case7 全複製 + `_build_accumulate` 冒頭に `world.step < ACCUMULATE_MIN_LAUNCH_STEP` ガード追加

## 仮説 (Hypothesis)

case7 が vs `baseline_v4` で大幅劣勢 (推定 ~10-25%) する主因は **t14 罠**: `accumulate_fire` mission が `ACCUMULATE_KNEE_SHIPS=60` 到達時点 (production 高 home なら t13-15 前後) で 60 ships を一斉発射し、敵反撃で大半を喪失。10戦集計 (`baseline_v9 vs baseline_v7`) で **trigger 確率 70% (7/10)、罠時の対 v9 勝率 0% (0/7)** を実証。

**Mechanism**: `_build_accumulate` を **`world.step >= ACCUMULATE_MIN_LAUNCH_STEP=30`** でガードすれば、序盤の罠を構造的に消せる。中盤以降の accumulate 機能は維持。case7 base 純粋 disadvantage (-17pp、罠なし条件) は残るが、**罠 trigger 70% を 0% にする** ことで vs v4 が +20pp 級改善 → ≥55% に到達できる。

## 既存コードの現状

- **新規 case 番号**: `case10` (free)、`baseline_v10` 未登録
- **Base**: `bot/pipeline/rulebase/case7/baseline/` 全資産 (BEAM=False default、STAY/accumulate/lookahead/opponent_model/missions/movements 全部)
- **修正対象関数**: `bot/pipeline/rulebase/case7/baseline/missions/stay.py:_build_accumulate` (line 360-463)
  - 現状: `if not cfg.ACCUMULATE_ENABLED or not world.my_planets: return ...` のみで step による制限なし
  - 改修案: `if world.step < cfg.ACCUMULATE_MIN_LAUNCH_STEP: return ...` を追加
- **既存 ACCUMULATE config**: `KNEE_SHIPS=60`, `SAFETY_SHIPS=4`, `MIN_TARGET_TURNS=15`, `MAX_TARGET_TURNS=60`, `MAX_HOLD_TURNS=12`
- **過去 iter の所見**:
  - case8 iter1/2/3 (case7 base + beam/filter) は 4 連敗、~30% で collapse
  - case9 (case4 base + filter) で base 切替効果 +10pp、filter 害 -10pp を分離確認
  - root cause 分析で t14 罠の構造を特定 (10戦集計 + stay.py code review)

## スコープ (Scope)

### 新規 case 構成

```
bot/pipeline/rulebase/case10/                        # case7 全複製
├── __init__.py
├── main.py                                          # case7 と同型、コメントを case10 に置換
├── README.md                                        # baseline_v10 概要
├── baseline/
│   ├── ...                                          # case7 全コピー
│   ├── core/config.py                               # ★ ACCUMULATE_MIN_LAUNCH_STEP=30 追加
│   └── missions/stay.py                             # ★ _build_accumulate 冒頭にガード 1 行追加
├── configs/                                         # case7 と同一
└── evaluation/                                      # case7 と同一

bot/src/dataset/selfplay/agents.py                   # `"baseline_v10": ...` 追加
bot/tests/pipeline/rulebase/case10/                  # case7 test を全複製 + step_guard test 1 件
bot/pyproject.toml                                   # case10/evaluation の B008/E501 ignore 追加
```

### config 追加

```python
# core/config.py
# t14 罠対策: 序盤 step では accumulate を発動しない (memory: project_case7_t14_trap)
ACCUMULATE_MIN_LAUNCH_STEP: int = 30
```

### 関数改修

```python
# missions/stay.py:_build_accumulate (line 374 直後に挿入)
def _build_accumulate(...):
    holds: dict[int, int] = {}
    fire_missions: list[Mission] = []
    held_ids: set[int] = set()
    if not cfg.ACCUMULATE_ENABLED or not world.my_planets:
        return holds, fire_missions, held_ids
    if world.step < cfg.ACCUMULATE_MIN_LAUNCH_STEP:    # ★ NEW
        return holds, fire_missions, held_ids
    ...
```

## 実装ステップ

1. **`cp -r bot/pipeline/rulebase/case7 bot/pipeline/rulebase/case10`** で全複製、`__pycache__` 除去
2. **`case10/__init__.py` / `main.py` / `README.md`** の case7 → case10 参照置換
3. **`baseline/core/config.py`** に `ACCUMULATE_MIN_LAUNCH_STEP: int = 30` を `ACCUMULATE_*` 群末尾に追加
4. **`baseline/missions/stay.py:_build_accumulate`** 冒頭の guard チェック直後に 1 ブロック追加 (上記 patch)
5. **`bot/src/dataset/selfplay/agents.py`** に `"baseline_v10": "pipeline.rulebase.case10.baseline.agent:agent"` 追加
6. **`bot/pyproject.toml`** に case10 の B008/E501 ignore 追加 (case4-9 と同パターン)
7. **`bot/tests/pipeline/rulebase/case10/`** を新規作成:
   - `__init__.py`
   - `test_baseline_agent.py` — case7 から複製 (smoke + valid action shape、env.run は @slow)
   - `test_accumulate.py` — case7 から複製 (既存の hold/fire 挙動の regression、step≥30 で実行)
   - `test_accumulate_step_guard.py` — 新規:
     - `world.step=29` で `_build_accumulate` が空 dict / 空 fire_missions を返すこと
     - `world.step=30` 以降は通常通り発動すること
     - `ACCUMULATE_MIN_LAUNCH_STEP=0` で完全に case7 等価動作になること
8. lint / format / mypy / pytest 緑、`dev/test-bot` 通過確認 (slow test 除く)

## 検証方法

### ローカル

```bash
# 高速ループ
uv run --directory bot pytest tests/pipeline/rulebase/case10 -m "not slow" -x --no-header -q

# 全 CI gate
dev/test-bot

# import 確認
uv run --directory bot python -c "from pipeline.rulebase.case10.baseline.agent import agent; print(agent)"

# submit dry-run (新 case 投入時の必須チェック)
uv run --directory bot python -m submit submit rulebase/case10 --dry-run --skip-validation -m "case10 dry-run"
```

### 性能評価 (2 段階 sweep)

#### Stage A — sweep 30/50/100 各 30戦 (~15分)

```bash
# step=30 (default), 50, 100 を 30戦ずつ vs v4
# 各 config は core/config.py を一時的に書き換えて実行 (or 環境変数 override)
for STEP in 30 50 100; do
  # ACCUMULATE_MIN_LAUNCH_STEP=$STEP に設定
  uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v10 \
    --mode 1v1 -n 30 --seed $((80000 + STEP)) --parallel 4 --no-save-replay
done
```

- 判定: 最高勝率の step を選び、Stage B で 100戦評価。最高勝率が <40% なら撤退

#### Stage B — best step で 100戦 (~10分)

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v10 \
    --mode 1v1 -n 50 --seed 81000 --parallel 4 --no-save-replay
uv run --directory bot python -m dataset run --agents baseline_v10,baseline_v4 \
    --mode 1v1 -n 50 --seed 81500 --parallel 4 --no-save-replay
```

- **対戦相手**: `baseline_v4` (production)
- **エピソード数**: 合算 100 戦 (seat0=50, seat1=50)、user 指定。memory `project_imitation_case1_phase3` は ≥300 推奨だが、本実験は +20pp 級の大効果を狙うため 100 戦で十分なシグナルが出る見込み
- **主要メトリクス**: 合算勝率 (vs v4)。Kaggle publicScore は使用しない
- **採否しきい値**: **+5pp 以上 (合算 ≥55%)** で採用。case7 (推定 ~25%) から +30pp の改善目標、罠 trigger を 0 に下げれば妥当
- **time budget**: turn_p95 ≤ 0.7s (BEAM=False、accumulate guard はほぼ noop コスト)
- **wall-clock 想定**: Stage A ~15分 + Stage B ~10分 = 計 25 分以内

### 副次評価 (smoke、Stage A 後の判断材料)

- **case10 vs case7 (10戦)** — t14 罠が消えたかを **直接観察**。10戦中 0 戦で self ship_loss_burst at t14 になることを replay で確認

## リスクと早期撤退条件

- **case7 base 純粋 disadvantage (-17pp) が支配項**: t14 罠を消しても vs v4 が 30-40% 止まり → 採用却下、case7 base に見切り
- **`ACCUMULATE_MIN_LAUNCH_STEP=30` が長すぎる**: 中盤の accumulate がほぼ発動せず、case7 の旨味が消える → step=20 などに緩めて再評価 (Stage A sweep でカバー)
- **fleet_consolidation 等の他 case4 mission との比較**: case10 が ≥55% に届いても case4 (LB745) を超えない可能性 — production 置換は別実験

## 期待される結果のシナリオ

| case10 vs v4 | 解釈 |
|---|---|
| ≥55% | **t14 罠が collapse 主因確定**、case7 base 改修で production 候補に上がる |
| 40-55% | 罠は確かに改善したが case7 純粋弱さが残る、case10 採用は微妙 |
| <40% | t14 罠以外にも case7 構造的弱点あり、heuristic 飽和 — case7 系見切り |
