# rulebase/case13 — predict_planet_position の precompute cache 化

> 作成日: 2026-05-06
> 関連 memory:
> - `project_case4_turn_p95_at_limit` — case4 greedy で turn_p95=0.678s、actTimeout 1.0s に対し margin 0.32s のみ
> - `project_case4_hot_path` — cProfile で CPU 96% が physics.py、predict_planet_position 56M call/game
> - `project_heuristic_search_saturation` — heuristic 探索 11 連敗、value head 注入が唯一残る道
> 関連 plan: `../20260504_case8_multistep_optimization/iter11_plan.md` (Action-level MCTS は本案で margin を作ってから着手)
> スコープ: case4 baseline を構造そのままで precompute cache を 1 箇所追加、turn_p95 を下げ value head 注入の余地を作る

## 仮説 (Hypothesis)

`physics.predict_planet_position(planet, initial, ang_vel, turns)` を **1 turn 開始時に NumPy で全 (planet × turn) を precompute → dict lookup に置換** すれば、勝率不変のまま turn_p95 を 0.678s → 0.5s 以下に下げられる。

**Why**: 関数は pure (副作用なし、入力同値 → 出力同値)。56M call の重複度は高い (planet 数 ~12 × turn offset 範囲 ~50 = 600 unique 入力 × callsite 約 56M / 600 ≈ 9 万回 hit/unique)。NumPy で一括 cos/sin 計算した結果を `(planet_id, turn) → (x, y)` 辞書に詰めれば、各 caller は dict.get で済む。**数値的に同一**なので勝率も完全に同一なはず — 検証は ±2pp/200戦 で抜け道なし。

## 既存コードの現状 (Step 1 から)

- `bot/pipeline/rulebase/case4/baseline/core/physics.py:36-53` — `predict_planet_position`, pure function, math.cos/sin/atan2 使用
- caller chain: `world_model.plan_shot → physics.aim_with_prediction → search_safe_intercept → _hit_turn_for_target_position → predict_target_position → predict_planet_position`
- 別 shadow copy: `core/safety.py:65 _predict_planet_position` (本 plan ではスコープ外、勝率不変検証後の iter2 で統合検討)
- WorldModel (`core/world_model.py:337`) は per-turn instance、initial_by_id / ang_vel を保持 → cache の自然な置き場

過去 iter で **速度最適化のみを hypothesis にした case は無し** (case5-12 はすべて勝率向上系)、よって case13 は新規 directory が妥当。

## スコープ (Scope)

- 変更ファイル:
  - `bot/pipeline/rulebase/case13/baseline/core/physics.py` — `predict_planet_position` を cache lookup ラッパに変更 (case4 全複製後)
  - `bot/pipeline/rulebase/case13/baseline/core/world_model.py` — WorldModel に `predicted_planet_pos: dict[tuple[int, int], tuple[float, float]]` フィールド追加、コンストラクタで precompute
- 不変ファイル: `safety.py` の shadow copy はスコープ外 (iter2 で扱う)
- ハイパーパラメータ: なし (純粋にロジック変更、外部設定なし)
- データセット / 特徴量: なし

## 実装ステップ (Implementation outline)

1. `cp -r bot/pipeline/rulebase/case4 bot/pipeline/rulebase/case13`
2. `case13/{main.py, __init__.py, README.md}` の case4 → case13 参照置換、README に「速度最適化のみ、勝率不変が成功条件」を 1 行記載
3. `case13/baseline/core/world_model.py` の WorldModel `__post_init__` または `build_world` 末尾で:
   - 静的 planet (`is_static_planet(p)`) は cache 不要 (位置不変)
   - 回転 planet について `turns ∈ [0, MAX_LOOKAHEAD_TURNS]` (例: 0..120) を NumPy で一括計算 → `predicted_planet_pos[(planet_id, turn)] = (x, y)` を埋める
   - MAX_LOOKAHEAD_TURNS は `_first_engine_hit_turn` の上限 turn と同期させる (cProfile 結果から実測)
4. `case13/baseline/core/physics.py:predict_planet_position` を以下に置換:
   ```python
   def predict_planet_position(planet, initial_by_id, angular_velocity, turns, *, cache=None):
       if cache is not None and (planet.id, turns) in cache:
           return cache[(planet.id, turns)]
       # 元の実装にフォールバック (cache 範囲外 turn / 静的 planet)
       ...
   ```
5. caller (`predict_target_position` など) に `cache=world.predicted_planet_pos` を伝播
6. `bot/src/dataset/selfplay/agents.py` に `"baseline_v13": "pipeline.rulebase.case13.baseline.agent:agent"` 追加
7. `bot/tests/pipeline/rulebase/case13/`:
   - `test_cache_equivalence.py` — `predict_planet_position` が cache 有無で同一値を返す (10 planet × 100 turn random)
   - `test_agent_smoke.py` — `env.run([agent, "random"])` が DONE する
8. lint / format / mypy / pytest 緑、`dev/test-bot` 通過

## 検証方法 (Validation method)

### ローカル

```bash
uv run --directory bot pytest tests/pipeline/rulebase/case13 -m "not slow" -x
dev/test-bot
```

### 性能 / 勝率評価 (3 段階、すべて Python simulator、rust 未 build なら Python fallback)

#### Stage A: 30戦 smoke (実装バグ検出)

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v13 \
    --mode 1v1 -n 30 --seed 300000 --parallel 4 --no-save-replay
```

判定:
- 勝率 50±10% かつ turn_p95 改善あり → Stage B
- 勝率 < 40% or > 60% → 実装バグ (cache 値ズレ等)、修正
- turn_p95 改善なし → cache hit してない、precompute 範囲を見直し

#### Stage B: 200戦 (本検証)

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v13 \
    --mode 1v1 -n 100 --seed 300100 --parallel 4 --no-save-replay
uv run --directory bot python -m dataset run --agents baseline_v13,baseline_v4 \
    --mode 1v1 -n 100 --seed 300300 --parallel 4 --no-save-replay
```

#### Stage C (option): 同条件で cProfile 再測定

```bash
uv run --directory bot python /tmp/profile-case13/run.py
```

`predict_planet_position` の tottime が 39s → 5s 以下、cumtime ベースで 70s → 10s 以下を確認。

### 採否しきい値

- **対戦相手**: `baseline_v4` (production, LB745)
- **エピソード数**: 合算 200 戦 (seat0=100, seat1=100)、user loop 指定
- **主要メトリクス**:
  1. **勝率変化 ≤ ±2pp** (合算 48-52% に収まる) — 挙動完全等価の検証
  2. **turn_p95 ≤ 0.5s** (case4 比 -25% 以上の改善) — 速度最適化の成功条件
- **Kaggle publicScore は使用しない** (project rule)
- 両条件を満たせば採用、`baseline_v13` を value head 注入の base に昇格

### リスクと早期撤退条件

- **cache 範囲外 turn の頻発で fallback 経由が大半 → tottime 改善せず** → MAX_LOOKAHEAD_TURNS を実測ベースで広げる (~150 turn)
- **dict lookup overhead が math.cos/sin 計算より高くつく** → `np.ndarray[planet_id × turn]` で 2D index access に変更
- **勝率乖離 > 2pp** → 浮動小数点丸め差で `_first_engine_hit_turn` の境界条件が変わった可能性、cache 値の dtype を float64 で揃え、初期 caller を 1 turn ずつ trace
- **shadow copy `safety._predict_planet_position` を統合しない選択** が原因で `safety.py` の hot path 分が改善しない可能性 → iter2 でスコープ拡張

## 参考 (References)

Step 3 web research は実施せず (caching は十分標準的、追加調査不要)。

## 進行管理

iter1 = `predict_planet_position` cache 化のみ。Stage A → B を順次実施。
両条件達成 → `baseline_v13` を新 production 候補にし、value head 注入 plan の input にする。
未達成 → iter2 で safety.py shadow copy 統合 or Numba JIT に拡張、再検証。
