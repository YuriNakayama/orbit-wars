# Rulebase/case10 — iter3: Phase 5 Thrash Suppression

> 作成日: 2026-05-05
> 関連:
> - [`iter1_plan.md`](./iter1_plan.md) / [`iter1_result.md`](./iter1_result.md) — step guard 単独で 53.0%
> - [`iter2_plan.md`](./iter2_plan.md) / [`iter2_result.md`](./iter2_result.md) — KNEE=40 は逆効果 (-3pp)
> - replay 分析: `data/output/experiment/rulebase/case10/replay_analysis/20260505_iter2/result_1.md` で Phase 5 (t90-150) の planet thrash 連鎖を確認
> スコープ: case10 上で `apply_score_modifiers` に「同 planet への capture/snipe/swarm 直近 N ターン commit 2 回以上で score 抑制」を追加

## 仮説 (Hypothesis)

iter2 の replay 分析で **Phase 5 (t90-150) の planet thrash 連鎖** が決定的敗因と判明:
- 同 planet (#19, #16, #2 など) を **取って→1-3 turn で奪われ→取り返しに行く** が連鎖
- 1 試合で planet#2 を 5 回奪取・5 回喪失 (case8 / case9 でも観測)

`apply_score_modifiers` で **「直近 N=8 ターンに自軍が同 planet 宛 capture/snipe/swarm を 2 回以上 commit した場合 score を 0.4 倍」** に減衰すれば、敵が安定して守れている planet への投資を抑制し、別 target に切り替わる。case8 iter3 v0 で類似実装が暴走したが、(a) **mission 種別を絞って記録** (b) **`apply_score_modifiers` 内で記録対象を限定** で正しく動作する見込み。

期待効果: thrash で消費していた ship を別 target に振り向けることで vs v4 で **+2-5pp 改善**、case10 を ≥55% に到達させる。

## 既存コードの現状

- iter1 の step guard で t14 罠は完全抑制済 (replay 確認)
- iter2 の KNEE=40 は逆効果、復元済 (KNEE=60)
- case8 iter3 v0 の bug: `_record_mission_commits` が move 種別を区別せず harass / accumulate / followup も記録 → `commits >= 2` が常に真化、filter 全 planet で誤発火
- case9 で iter3 v1 は recently_lost only に切替、害は出なかったが効果薄

## スコープ (Scope)

### 変更ファイル (3 ファイル)

```
bot/pipeline/rulebase/case10/baseline/
├── core/config.py                 # ★ THRASH_REPEAT_* config 追加 (新規 4 個)
├── core/world_model.py            # ★ mission_commits 引数追加 (case9 と同型)
├── agent.py                       # ★ StayState に mission_commits 追加 + 記録ロジック
└── strategy_helpers.py            # ★ apply_score_modifiers に decay 追加
```

### config 追加

```python
# core/config.py
THRASH_REPEAT_FILTER_ENABLED: bool = True
THRASH_REPEAT_WINDOW: int = 8        # 直近 N ターン
THRASH_REPEAT_LIMIT: int = 2         # commit 回数しきい値
THRASH_REPEAT_SCORE_MULT: float = 0.4 # score 倍率
```

### Thrash 判定ロジック (case8 iter3 v0 のバグを正しく修正)

case8 v0 の bug: `_record_mission_commits` が move から target を逆推論し、harass / accumulate も記録。
本実装の**正しい記録方法**:
1. `_process_single_source_mission` 内で mission を commit する直前に **`mission.kind in ("capture","snipe","swarm")` のときだけ** `_STAY_STATE.mission_commits[target_id]` に append
2. 記録には move emit 後の `mission.target_id` を直接使う (逆推論しない)
3. `apply_score_modifiers` で `mission in ("capture","snipe","swarm")` && `commit_count >= THRASH_REPEAT_LIMIT` のとき score を 0.4 倍

これで bug 1 (種別誤記録)、bug 2 (逆推論誤判定) ともに解消。

## 実装ステップ

1. `core/config.py` に THRASH_REPEAT_* 4 個を追加
2. `core/world_model.py` に `mission_commits: dict[int, list[int]] | None = None` 引数追加 (case9 と同型 1 行)
3. `agent.py:StayState` に `mission_commits: dict[int, list[int]]` フィールド追加
4. `agent.py` に `_record_mission_commit_target(planet_id: int, step: int)` 関数追加 — `_STAY_STATE.mission_commits[planet_id].append(step)` だけ。pruning は per-turn 1 回 `_update_thrash_state` 相当で
5. `strategy.py:_process_single_source_mission` の commit 成功直後 (move append 後) に **mission.kind in ("capture","snipe","swarm")** のとき `_record_mission_commit_target` 呼び出し
6. `strategy_helpers.py:apply_score_modifiers` に block 追加: capture/snipe/swarm の score を `world.mission_commits` に基づき 0.4 倍
7. `agent.py:agent` で `mission_commits` を `dict(_STAY_STATE.mission_commits)` で WorldModel に渡す
8. tests/pipeline/rulebase/case10/test_thrash_repeat_filter.py を新規追加

## 検証方法

### ローカル

```bash
uv run --directory bot pytest tests/pipeline/rulebase/case10 -m "not slow" -x
```

### 性能評価 (30戦, user 指定)

```bash
uv run --directory bot python -m dataset run --agents baseline_v4,baseline_v10 \
    --mode 1v1 -n 30 --seed 83000 --parallel 4
```

判定:
- ≥55% → iter3 採用候補、Stage B 100戦で確認推奨
- 53-55% → iter1 (53.0%) と差なし、撤退
- <50% → repeat filter は逆効果、撤退

## リスクと早期撤退条件

- **case8 iter3 v0 と同じ bug の再発**: mission 種別を `_process_single_source_mission` から取得するので問題なし、ただし `_process_multi_source_mission` (swarm) も忘れず対応する
- **filter が agressive 過ぎて全攻撃 mission 抑制**: replay で「他に target がない」局面で thrash を続けるしかない場合がある → ≥40 は確保、 1 試合で 0 attack にはならない
- **case9 で確認済の害 (-10pp)**: case9 は recently_lost only で害が出た。本実験は repeat 経路 + 限定的な発動 (window=8 短い) なので、case9 の害とは別性質
