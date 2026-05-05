# Rulebase/case8 — Planet Thrash Filter (iter3)

> 作成日: 2026-05-05
> 関連:
> - [`iter1_result.md`](./iter1_result.md) — beam (legacy_net_ships) で 32.3% 却下
> - [`iter2_result.md`](./iter2_result.md) — beam (mission_score + true2p_sampled) で 27% 却下、撤退
> - replay 分析 (iter2): `data/output/experiment/rulebase/case8/replay_analysis/20260505_1410_iter2/result_{1,2}.md` — planet thrash 連鎖を実証
> - [`docs/experiment/rulebase/20260420_case3_rollout_ablation/result.md`](../20260420_case3_rollout_ablation/result.md) — 「次の改善は候補生成置換 / 学習評価関数」
> スコープ: case8 内 (新 case を切らない)、`BEAM_ENABLED=False` のまま greedy 経路上で mission score を加工

## 仮説 (Hypothesis)

直近 N=10 ターンに **(a) 自軍が他 owner に奪われた planet**、または **(b) 自軍が同 planet 宛に mission を 2 回以上 commit した planet** に対する **capture / snipe / swarm mission の score を 0.3 倍に減衰** すると、replay 分析で確認した planet thrash 連鎖 (planet#2 を 32 ターンに 5 回奪取・5 回喪失 / planet#21 を 16 ターンに 5 回 thrash) を構造的に抑制でき、vs `baseline_v4` (production) のローカル 200戦勝率を **+5pp 以上 (≥55%)** 改善できる。

**Mechanism**:
- 現状の greedy `mission.score` は「**目先の奪取価値**」だけ見て「**直後の奪還リスク**」を無視 → 1-2 ターン後に奪い返される planet にも全力 ship を発射し、攻撃成功 → 即奪還 → 再攻撃 のループで ship を浪費
- thrash filter で同 planet への score を抑えれば、greedy が他の安定 planet (奪われた履歴のない neutral / 確実に守れる位置) を優先選択 → 浪費が止まる
- iter1/iter2 の 「heuristic を beam で取り囲む」方針 (= score の上に探索層を載せる) が飽和したのに対し、本案は **heuristic score 自体を妥当な方向に修正** する第一歩。case3 result.md が示した第 1 番目の未踏領域 (候補生成置換) に該当

## 既存コードの現状

- **base**: `bot/pipeline/rulebase/case8/baseline/` (= case7 全複製、iter1/2 の beam 資産は残置、`BEAM_ENABLED=False` で case7 等価動作)
- **mission builders**: `missions/{capture,snipe,swarm,harass,reinforcement,crash_exploit,stay}.py` — capture/snipe/swarm が thrash 主因
- **score 集約点**: `strategy_helpers.score_attack` → `apply_score_modifiers(base, target, mission, world)` で mission 種別 modifier を適用 (`STATIC_TARGET_SCORE_MULT`, `SNIPE_SCORE_MULT` 等) — **ここに 1 行追加すれば全 attack mission に効く**
- **per-game 状態**: `agent.py:StayState` (consecutive_holds / accumulate_holds / last_step) — `recently_lost` / `mission_commit_history` を同 dataclass に追加可能
- **owner 変化検出のリファレンス実装**: case3 の `CaptureState` (`pipeline/rulebase/case3/baseline/core/world_model.py`) で REINFORCE_FRESH_CAPTURE 用に owner 遷移を記録している。アルゴリズムを参考にできる (cross-case import 禁止のため複製のみ)

## スコープ (Scope)

### 変更ファイル (2 ファイルのみ)

```
bot/pipeline/rulebase/case8/baseline/
├── agent.py                  ★ StayState に recently_lost / mission_commits 追加、毎ターン更新
└── strategy_helpers.py       ★ apply_score_modifiers に thrash decay 1 ブロック追加
```

加えて:
```
bot/pipeline/rulebase/case8/baseline/core/
└── config.py                 ★ THRASH_* config 追加 (3 個)
└── world_model.py            ★ recently_lost / mission_commits を WorldModel に渡せるよう 2 引数追加
```

実装規模感: 計 ~50-80 行の追加。

### config 追加

| 名前 | 値 | 役割 |
|---|---|---|
| `THRASH_FILTER_ENABLED` | `True` | OFF で iter1/2 と同等動作 (回帰テスト用) |
| `THRASH_WINDOW` | `10` | 直近何ターンの履歴を「thrash 中」と見なすか (case3 REINFORCE_FRESH_CAPTURE_WINDOW=10 と整合) |
| `THRASH_SCORE_MULT` | `0.3` | thrash 認定 planet への capture/snipe/swarm score の倍率 (`STATIC_TARGET_SCORE_MULT` 等と同オーダー) |
| `THRASH_REPEAT_COMMIT_LIMIT` | `2` | 同 planet 宛 mission を window 内に何回 commit したら「点従品」と判定するか |

### Thrash 判定ロジック

planet `p` が **直近 `THRASH_WINDOW` ターン以内に下記のいずれかを満たす** とき thrash 認定:
- (a) **owner 遷移**: `self → 他 owner` (-1 含む) の遷移が 1 回以上発生
- (b) **mission 過剰投資**: 自軍が `p` 宛 mission (capture/snipe/swarm のみカウント) を `THRASH_REPEAT_COMMIT_LIMIT=2` 回以上 commit した

(a) が replay 仮説の「奪われ planet を即奪い返しに行く」抑制、(b) が「奪い返しに行く前に過剰投資パターン」抑制。

decay は capture / snipe / swarm の 3 mission 種別にのみ適用 (reinforce / harass / crash_exploit / accumulate には適用しない)。

## 実装ステップ

1. **`core/config.py`** に `THRASH_*` 4 個を追加。
2. **`agent.py:StayState`** に `recently_lost: dict[planet_id, last_loss_turn]` と `mission_commits: dict[planet_id, list[turn]]` を追加。同 dataclass で per-game 状態を一括管理。
3. **`agent.py:_reset_stay_state_if_new_episode`** で新規 episode 検出時に上記 2 dict も clear。
4. **`agent.py:agent`** で:
   - `build_world` 後に **owner 遷移を検出** (`prev_planets` を保持し、`p.owner` の `self → other` 遷移で `recently_lost[p.id] = world.step`)
   - `plan_moves` 戻り値の `moves` から **mission target を逆推論** (move の angle から最近接 planet を取り、`mission_commits[target_id].append(world.step)`)
   - 古いエントリ (turn 差 > THRASH_WINDOW) を pruning
5. **`core/world_model.py:WorldModel.__init__`** に `recently_lost: dict[int, int] | None = None` と `mission_commits: dict[int, list[int]] | None = None` の 2 引数追加 (`{}` default、`base_timeline` 計算には影響しない)。
6. **`strategy_helpers.apply_score_modifiers`** で:
   ```python
   if THRASH_FILTER_ENABLED and mission in ("capture", "snipe", "swarm"):
       step = world.step
       lost_at = world.recently_lost.get(target.id)
       commit_count = sum(
           1 for t in world.mission_commits.get(target.id, [])
           if step - t <= THRASH_WINDOW
       )
       is_thrash = (
           (lost_at is not None and step - lost_at <= THRASH_WINDOW)
           or commit_count >= THRASH_REPEAT_COMMIT_LIMIT
       )
       if is_thrash:
           score *= THRASH_SCORE_MULT
   ```
7. **`agent.py:agent`** で `build_world` 呼び出しに `recently_lost=_STAY_STATE.recently_lost, mission_commits=_STAY_STATE.mission_commits` を渡す。
8. **`tests/pipeline/rulebase/case8/test_thrash_filter.py`** を新規追加 — `WorldModel` に `recently_lost={target_id: step-3}` を渡し、`score_attack(...)` の戻り値が thrash 認定で 0.3 倍になることを確認。
9. **`tests/pipeline/rulebase/case8/test_beam_off_equals_greedy.py`** は **再評価**: thrash filter は greedy にも効く (= case7 と差が出る) ので、`THRASH_FILTER_ENABLED=False` で監視するように更新。または「filter OFF + BEAM OFF で case7 等価」を新たな保証に変更。
10. lint / format / mypy / pytest 緑、`dev/test-bot` 通過を確認。

## 検証方法

### ローカル

```bash
# 高速ループ
uv run --directory bot pytest tests/pipeline/rulebase/case8 -m "not slow" -x --no-header -q

# 全 CI gate
dev/test-bot
```

### 性能評価 (2 段階)

#### Stage A — smoke / log 分析 (50戦)

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v8 \
    --mode 1v1 -n 50 --seed 50000 --parallel 4
```

- 結果勝率を確認 + replay 5戦前後を **Phase 1 自動 pick (長戦+最速敗北)** で抽出 → `replay_to_markdown.py` で thrash 件数の前後比較
- 判定: vs v4 ≥40% かつ replay の planet thrash 連鎖が顕著に減っていれば Stage B へ進む。それ未満なら hyperparameter (THRASH_WINDOW / THRASH_SCORE_MULT) を 1-2 段振ってから再評価

#### Stage B — 200戦評価

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v8 \
    --mode 1v1 -n 100 --seed 50000 --parallel 4 --no-save-replay
uv run --directory bot python -m dataset run --agents baseline_v8,baseline_v4 \
    --mode 1v1 -n 100 --seed 50500 --parallel 4 --no-save-replay
```

- **対戦相手**: `baseline_v4` (production)
- **エピソード数**: 合算 200戦 (seat0=100, seat1=100、user 指示)
- **主要メトリクス**: 合算勝率 (vs v4)。**Kaggle publicScore は使用しない**
- **採否しきい値**: **+5pp 以上 (合算 ≥55%)** で採用。iter1=32.3% → iter3 で +22pp の改善目標
- **time budget**: turn_p95 ≤ 0.7s (`BEAM_ENABLED=False` のため iter1 並み 0.05-0.10s 予測、十分に余裕)
- **wall-clock 想定**: 200戦合算 ~10 分以内 (BEAM 撤回でiter1並み)

### Tuning (Stage A 結果次第で実施)

| 構成 | 期待 |
|---|---|
| `THRASH_WINDOW = 10`, `THRASH_SCORE_MULT = 0.3` (default) | replay 仮説と整合する基本構成 |
| `THRASH_WINDOW = 5`, mult 0.5 (より軽め) | filter が強すぎて攻撃 mission 全否定にならないか |
| `THRASH_WINDOW = 15`, mult 0.2 (より強め) | thrash 完全抑制、ただし「奪い返したい」局面も逃す可能性 |

## リスクと早期撤退条件

- **filter が攻撃 mission 全否定**: thrash 認定 planet が多すぎて capture/snipe/swarm が常に減衰 → 何も攻撃しない agent になる。Stage A で 0/50 や 5/50 のような極端な敗北なら hyperparameter 緩和
- **owner 遷移検出のバグ**: comet planet (出現/消滅) や initial_planet でない planet で owner が `-1 → -1` などの edge case を誤判定。test で確認
- **mission target 逆推論の精度**: `move = [src_id, angle, ships]` から target を逆推論する処理は angle が他 planet を狙っていない場合 (空打ち) もある。`opponent_reaction.py:_infer_action_target` を流用して精度確保
- **計算予算超過**: filter 自体は O(planets × THRASH_WINDOW) で軽量、apply_score_modifiers の hot path に入るが per-call <0.1ms 想定。turn_p95 への影響はほぼなし

## 関連ファイル (実装後に作成 / 更新)

- `bot/pipeline/rulebase/case8/baseline/core/config.py` — THRASH_* 4 個追加
- `bot/pipeline/rulebase/case8/baseline/agent.py` — StayState 拡張、owner 遷移 / mission commit 追跡
- `bot/pipeline/rulebase/case8/baseline/core/world_model.py` — recently_lost / mission_commits 引数追加
- `bot/pipeline/rulebase/case8/baseline/strategy_helpers.py:apply_score_modifiers` — thrash decay 追加
- `bot/tests/pipeline/rulebase/case8/test_thrash_filter.py` — 新規
- `bot/tests/pipeline/rulebase/case8/test_beam_off_equals_greedy.py` — `THRASH_FILTER_ENABLED=False` でも等価性を保証するよう更新
