# case7 Pool 形式 RL — アーキテクチャ設計

## 全体図

```
                       ┌──────────────── train_jax.py (host, Python) ────────────────┐
                       │                                                              │
  config (pool yaml) → │  _build_model → (BC) → _maybe_resume                         │
                       │                                                              │
                       │  iter loop:                                                  │
                       │    _OpponentPool.push(model)   every snapshot_every          │
                       │    _PrioritizedOpponentSelector.select()                     │
                       │      entries = [full, CASE8(new)] + past-self snapshots       │
                       │      weight = (1-win_ema)^p, exploiter_prob_cap で full+case8 │
                       │      合計確率を上限クリップ → 超過分は past-self へ再配分       │
                       │           │ opponent_mode:int, opp_model                      │
                       │           ▼                                                  │
                       │    collect_rollout_jax ───────────────┐                      │
                       │    ppo update                          │                      │
                       │    ▼ (iter 末)                          │                      │
                       │    _eval_in_jax(model, eval_opponent, N) ← NEW               │
                       │      eval_win >= best → _save_best_pt(best.pt)                │
                       │    _save_best_pt(ckpt_i{NNN}.pt)  毎iter                      │
                       │    _write_metrics(eval_win, exploiter_sel_rate, ...)         │
                       └──────────────────────────────────────────┼──────────────────┘
                                                                   ▼
                       ┌──────────── rollout_jax.py (pure JAX, vmap/scan) ────────────┐
                       │  _rollout_one_env:                                           │
                       │    opp_*_actions = [noop, lite, full, snapshot, v1,v4,v8,     │
                       │                     CASE8(new mode 7)]                        │
                       │    CASE8 = compute_actions(                                   │
                       │             build_world_features_from_state(state,1-seat),    │
                       │             _modes_from_features(...))   ← in-JAX, host hop無 │
                       │    opp_actions = lax.switch(opponent_mode, [...])             │
                       │    + _shaping_potentials (ratio/1.0, 変更なし)                │
                       └──────────────────────────────────────────────────────────────┘
```

## モジュール設計(変更点のみ)

### rollout_jax.py(pure JAX 層)
- **新 import**:
  ```python
  from pipeline.rulebase.case8.baseline_jax.agent_jax import compute_actions as _case8_compute, _modes_from_features as _case8_modes
  from pipeline.rulebase.case8.baseline_jax.world_features import build_world_features_from_state as _case8_features
  ```
- **新 action fn**:
  ```python
  def _baseline_case8_actions(state: EnvState, seat: int) -> jax.Array:
      feats = _case8_features(state, seat)
      modes = _case8_modes(feats)
      return _case8_compute(feats, modes)  # (L,3)
  ```
- **mode 定数追加**: `OPPONENT_BASELINE_JAX_CASE8 = 7`、`OPPONENT_NAME_TO_MODE["baseline_jax_case8"] = 7`。
- **`lax.switch` を 0-7 に拡張**: `jnp.clip(opponent_mode, 0, 7)` + branch 追加。
- **shaping は無変更**。

### train_jax.py(host 層)
- **`_PrioritizedOpponentSelector`**:
  - `__init__` の固定 entry を config 駆動に(`exploiters: ["baseline_jax_full","baseline_jax_case8"]`)。
  - `select()` 後、exploiter(full+case8)合計選択確率が `exploiter_prob_cap` 超なら past-self へ再配分。
  - `rebuild()` の carry-over を exploiter 種別ごとに index 整合。
- **`_eval_in_jax(model, opponent_name, episodes, key) -> float`**(新規):
  既存 `collect_rollout_jax`(or 専用軽量 rollout)を `opponent=eval_opponent` で N 回し、
  terminal reward 符号から win_rate を算出。argmax 決定論で。in-JAX、host hop 無し。
- **iter loop**: `_eval_in_jax` を毎 iter 末に呼び、`eval_win` で best.pt を gate
  (現状の self-play `win_rate>=best` を `eval_win>=best_eval` に置換、S3 upload も連動)。
- **`_write_metrics`**: row に `eval_win`/`eval_opponent`/`exploiter_sel_rate` 追加。

### configs/pool_default.yaml(新規・正典)
```yaml
training:
  horizon: 500            # 必須
  shaping_mode: ratio
  shaping_coef: 1.0
  opponent: curriculum
  opponent_curriculum: {switch_iter: 4, early: noop, late: pool}
  opponent_pool:
    snapshot_every: 4
    cap: 4
    priority: f_hard
    priority_p: 2.0
    priority_ema: 0.7
    exploiters: [baseline_jax_full, baseline_jax_case8]   # NEW
    exploiter_prob_cap: 0.2                                # NEW
  eval:                                                    # NEW
    eval_opponent: baseline_jax_case8
    eval_episodes: 6
    select_metric: eval_win
```

## データモデル
- **metrics.json row**(追記): `eval_win:float`, `eval_opponent:str`, `exploiter_sel_rate:float`。
- **run_dir**: `best.pt`(npz, eval_win gate), `ckpt_i{NNN}.pt`(毎iter), `metrics.json`, `run.json`。
- 既存スキーマと後方互換(新 field は optional 読み出し)。

## ファイル配置(case 独立規約準拠)
| 変更 | ファイル | submission? |
|---|---|---|
| case8 opponent action fn + mode | `case7/training/rollout_jax.py` | No(.submitignore: training/) |
| selector + in-JAX eval + metrics | `case7/training/train_jax.py` | No |
| 正典 config | `case7/configs/pool_default.yaml` | No(.submitignore: configs/) |
| README 更新(pool 構成表) | `case7/README.md` | Yes(本文のみ) |
| テスト | `bot/tests/pipeline/reinforce/case7/` | No |

`training/` は submission 対象外なので `pipeline.rulebase.case8.*` の絶対 import 可
(Kaggle 実行されない)。case8 のコードは複製せず import で再利用。

## 外部統合
- 新規外部サービス無し。case8(既存内部)+ orbit_wars_jax(既存 sim)のみ。
- GPU 段階拡大時は `dev/runpod train`(既存基盤、`--case case7` 相当の配線確認)。
