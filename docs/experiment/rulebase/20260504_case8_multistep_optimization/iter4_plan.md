# Rulebase/case9 — Thrash Filter on case4 base

> 作成日: 2026-05-05
> 関連:
> - [`docs/experiment/rulebase/20260504_case8_multistep_beam/iter3_result.md`](../20260504_case8_multistep_beam/iter3_result.md) — case8 iter1/2/3 が ~30% で collapse、case7 base 起因の handicap 仮説
> - [`docs/experiment/rulebase/20260504_case8_multistep_beam/iter3_plan.md`](../20260504_case8_multistep_beam/iter3_plan.md) — thrash filter 詳細仕様 (流用元)
> - replay 分析 (case8 iter3 v1): `data/output/experiment/rulebase/case8/replay_analysis/20260505_1600_iter3v1/` — filter 機能確認 (thrash 187→45)
> - case4 README: `bot/pipeline/rulebase/case4/README.md` (LB745 production、`project_case4_phase_results.md` で 70.3% vs v3)
> スコープ: 新規 `case9` を切り、case4 (production) 全複製 + case8 iter3 v1 thrash filter 移植

## 仮説 (Hypothesis)

case8 iter1/2/3 が vs `baseline_v4` で ~30% に collapse した主因は **case7 base 自体の弱さ** (smoke で v4 vs v7 = 60-40)。case4 (production, LB745) を base に thrash filter を被せれば、base 起因の handicap (~10pp) が消えて vs v4 で **≥55% (+5pp)** を達成できる。同時に「case7 base が iter1/2/3 collapse の主因だった」ことを実証できる。

**Mechanism**:
- case4 vs case4 self-play は理論上 ~50% (base が同じ)
- そこに case8 iter3 v1 で実証された thrash filter (replay で thrash 件数 187→45 = 76% 削減) を加えれば、 +5pp 程度の小幅改善が期待
- case7 base の構造的 handicap (accumulate/STAY が production case では裏目に出る可能性) を排除した上で、filter の純粋効果を測れる

## 既存コードの現状 (from Step 1)

- **新規 case 番号**: `case9` (free slot、`AGENT_REGISTRY` も `baseline_v8` まで)
- **Base**: `bot/pipeline/rulebase/case4/baseline/` (LB745 production、case3 + fleet_consolidation)
  - 構造: `agent.py` + `strategy.py` (225行) + `strategy_helpers.py` (403行) + `missions/{capture,snipe,swarm,harass,reinforcement,crash_exploit}` + `lookahead.py` + `opponent_model.py`
  - case7/8 にあって case4 に **無い** もの: `stay/accumulate` mission、`planner/`、`StayState` クラス
  - case4 に **ある** が case7/8 に無いもの: `fleet_consolidation.py` mission、`physics.predict_target_position_fractional` + `SAFE_INTERCEPT_HALF_STEP`
- **score 集約点**: `strategy_helpers.apply_score_modifiers` (case8 と同型、thrash decay コピペ可能)
- **case8 iter3 v1 から再利用する資産**: `THRASH_*` 4 config / `_update_thrash_state` / `apply_score_modifiers` の thrash decay block / `test_thrash_filter.py`
- **過去 iter の所見**: case8 iter3 v1 で thrash 件数 76% 削減 (replay で実証) するも win_rate 30% に留まる → case7 base 起因が支配的という仮説

## スコープ (Scope)

### 新規 case 構成

```
bot/pipeline/rulebase/case9/                                # case4 全複製
├── __init__.py
├── main.py                                                  # case4 と同型、コメントを case9 に置換
├── README.md                                                # baseline_v9 概要
├── baseline/
│   ├── __init__.py
│   ├── agent.py                                             # case4 から複製 + StayState 追加 (thrash 用 minimal)
│   ├── strategy.py                                          # case4 と同一 (BEAM 不要)
│   ├── strategy_helpers.py                                  # ★ apply_score_modifiers に thrash decay 追加
│   ├── opponent_model.py                                    # case4 と同一
│   ├── lookahead.py                                         # case4 と同一
│   ├── core/
│   │   ├── config.py                                        # ★ THRASH_* 4 個追加
│   │   └── world_model.py                                   # ★ recently_lost / mission_commits 引数追加
│   ├── missions/                                            # case4 と同一 (capture/snipe/swarm/harass/reinforcement/crash_exploit/fleet_consolidation)
│   └── movements/                                           # case4 と同一
├── configs/                                                 # case4 と同一 (.submitignore 既存パターンで除外)
└── evaluation/                                              # case4 と同一 + compare_v4.py 新規

bot/src/dataset/selfplay/agents.py                           # `"baseline_v9": ...` 追加
bot/tests/pipeline/rulebase/case9/                           # 新規、case8 thrash filter test を流用
bot/pyproject.toml                                           # case9/evaluation の B008/E501 ignore 追加 (case4-8 と同パターン)
```

### config 追加 (case8 iter3 v1 と同一)

```python
THRASH_FILTER_ENABLED: bool = True
THRASH_WINDOW: int = 10
THRASH_SCORE_MULT: float = 0.3
THRASH_REPEAT_COMMIT_LIMIT: int = 999  # commits 経路は無効化、recently_lost only
```

### Thrash 判定ロジック

case8 iter3 v1 と同一:
- planet `p` が直近 `THRASH_WINDOW=10` ターン以内に **self → 他 owner** の遷移をした場合 thrash 認定
- `THRASH_REPEAT_COMMIT_LIMIT=999` のため mission_commits 経路は実質無効
- `apply_score_modifiers` 内で `mission in ("capture", "snipe", "swarm")` のときのみ score を 0.3 倍

## 実装ステップ

1. **`cp -r bot/pipeline/rulebase/case4 bot/pipeline/rulebase/case9`** で全複製。
2. **`case9/__init__.py` / `main.py` / `README.md`** の case4 → case9 参照置換 (Kaggle entrypoint コメント / ドキュメント)。
3. **`baseline/core/config.py`** に `THRASH_*` 4 個を末尾に追加 (case8 iter3 v1 と同一値、コメントも追記)。
4. **`baseline/core/world_model.py:WorldModel.__init__`** に `recently_lost: dict[int, int] | None = None` と `mission_commits: dict[int, list[int]] | None = None` の 2 引数追加 (case8 と完全同型)。
5. **`baseline/agent.py`** に `StayState` (thrash 用 minimal: `last_step` / `prev_planet_owners` / `recently_lost` / `mission_commits` のみ)、`_reset_stay_state_if_new_episode`、`_update_thrash_state` を追加。case8 iter3 v1 から複製、STAY/accumulate 関連は除外 (case4 base に元々無い)。
6. **`baseline/agent.py:agent`** で `_reset_stay_state_if_new_episode` → `_update_thrash_state` → `build_world` (recently_lost / mission_commits を渡す) → `plan_moves` の順に呼び出すように改修。
7. **`baseline/strategy_helpers.py`** に `THRASH_*` import + `apply_score_modifiers` に thrash decay block 追加 (case8 と完全同型)。
8. **`bot/src/dataset/selfplay/agents.py`** に `"baseline_v9": "pipeline.rulebase.case9.baseline.agent:agent"` 追加。
9. **`bot/pyproject.toml`** に `"pipeline/rulebase/case9/evaluation/**/*.py" = ["B008", "E501"]` と `"tests/pipeline/rulebase/case9/snapshots/**" = ["ALL"]` を追加 (case4-8 と同パターン)。
10. **`bot/tests/pipeline/rulebase/case9/`** を新規作成:
    - `__init__.py`
    - `test_baseline_agent.py` — case8 のものを base に複製、smoke 用 unit test (slow integration mark で env.run も)
    - `test_thrash_filter.py` — case8 のものを完全複製、import path だけ case9 に
    - `test_filter_off_equals_base.py` — `THRASH_FILTER_ENABLED=False` で `baseline_v9` の moves が `baseline_v4` と一致することを保証 (case8 の `test_beam_off_equals_greedy.py` の case9 版)
11. lint / format / mypy / pytest 緑、`dev/test-bot` 通過確認 (slow test 除く)。

## 検証方法

### ローカル

```bash
# 高速ループ
uv run --directory bot pytest tests/pipeline/rulebase/case9 -m "not slow" -x --no-header -q

# 全 CI gate
dev/test-bot

# import 確認
uv run --directory bot python -c "from pipeline.rulebase.case9.baseline.agent import agent; print(agent)"

# submit dry-run (新 case 投入時の必須チェック)
uv run --directory bot python -m submit submit rulebase/case9 --dry-run --skip-validation -m "case9 dry-run"
```

### 性能評価

```bash
# 200戦 (seat 入替: 100×2)
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v9 \
    --mode 1v1 -n 100 --seed 60000 --parallel 4 --no-save-replay
uv run --directory bot python -m dataset run --agents baseline_v9,baseline_v4 \
    --mode 1v1 -n 100 --seed 60500 --parallel 4 --no-save-replay
```

- **対戦相手**: `baseline_v4` (production, LB745) — 主軸かつ base case (= 自己対戦に近い)
- **エピソード数**: 合算 200戦 (seat0=100, seat1=100)、user 指定
- **主要メトリクス**: 合算勝率 (vs v4)。**Kaggle publicScore は使用しない**
- **採否しきい値**: **+5pp 以上 (合算 ≥55%)** で採用。case4 vs case4 は理論 ~50%、+5pp で「thrash filter が production base にも価値を加えた」と確認できる
- **time budget**: turn_p95 ≤ 0.7s (BEAM なし、filter は O(planets × THRASH_WINDOW) で軽量、case4 並み 0.05-0.10s 想定)
- **wall-clock 想定**: 200戦合算 ~10 分以内 (case4 は case7 より高速、case8 iter3 v1 と同等)

### 副次評価 (smoke)

- 30戦 vs `baseline_v8` (case7 base + thrash filter) — base 切替の純粋効果を見る (期待: case9 が case8 を ≥10pp 上回る、これにより「base 起因の collapse」仮説を実証)

## リスクと早期撤退条件

- **Self-play すぎて差が出ない**: vs v4 結果が ~50% ±2pp に収束 → thrash filter は production base では効果なし、しかし case7 base 起因の collapse 仮説は確認できる (= 副次目的達成)
- **Filter が production case4 で害**: vs v4 = 40% 台に低下 → filter が case4 の delicate な balance を崩している。case8 iter3 v1 では機能した filter が case4 では別の behavior を引き起こす可能性
- **fleet_consolidation との相互作用**: case4 独自の fleet_consolidation mission は capture-like、thrash filter で score 減衰すると fleet 集約が弱まる可能性。要 replay で検証
- **submit shape**: 新 case のため `--dry-run --skip-validation` で必ず確認、validator (`__file__` 不使用 patterns) も Path.cwd() のままで OK

## (参考) iter3 result.md の優先 1 案として位置づけ

`docs/experiment/rulebase/20260504_case8_multistep_beam/iter3_result.md` で次の方向として 3 案提示:
1. **base を case4 に切り替え** (本実験) — 1 時間、case7 base 起因の handicap を切り分け
2. 序盤 (t<20) の attack mission 抑制 — 30 分、t14 ship 枯渇の構造修正
3. 学習評価関数 — 数日

本 plan は **(1) を実施** することで他案の判断材料を提供する。case9 で +5pp 達成 → case7 base が主因確定 → next iter で case4 base 上で他改善案を試す。case9 で 50% ±2pp → filter 効果限定 / case7 base 仮説は確認 → next iter で (2) や (3) に進む判断材料となる。
