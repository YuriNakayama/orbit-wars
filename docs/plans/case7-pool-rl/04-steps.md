# case7 Pool 形式 RL — 実装ステップ

実装順序方針: **rollout(in-JAX 相手) → host(selector/cap) → eval 統合 → config/テスト → GPU**。
依存が下流に流れる bottom-up。各ステップは 1 技術関心 = 1 タスク。

---

## Step 1: case8 を in-JAX opponent として rollout に追加
**Target**: training(rollout_jax.py) / **Dependencies**: None

### Overview
case8 の本物 parity JAX rule を `lax.switch` の新 mode(7)として追加。host hop 無し。

### Work Items
- [ ] `_baseline_case8_actions(state, seat)` を `compute_actions(build_world_features_from_state(state,seat), _modes_from_features(...))` で実装
- [ ] `OPPONENT_BASELINE_JAX_CASE8 = 7` 定数 + `OPPONENT_NAME_TO_MODE["baseline_jax_case8"] = 7`
- [ ] `lax.switch` を `clip(mode,0,7)` + branch 8 個に拡張
- [ ] vmap/scan friendly(int mode 維持、shape 固定)を確認

### Target Files
- `bot/pipeline/reinforce/case7/training/rollout_jax.py`

### Acceptance Criteria
- `opponent="baseline_jax_case8"` で 1 game rollout が完走、reward 符号が出る
- `ruff`/`mypy` clean、既存 mode(0-6)の挙動不変

---

## Step 2: selector に case8 entry + 混入率キャップ
**Target**: training(train_jax.py) / **Dependencies**: Step 1

### Overview
`_PrioritizedOpponentSelector` の固定 exploiter を config 駆動にし、full+case8 の合計
選択確率に `exploiter_prob_cap` 上限を設ける(超過分は past-self へ再配分)。

### Work Items
- [ ] `__init__`/`rebuild` の固定 entry を `exploiters: list[str]` 駆動に
- [ ] `select()` で exploiter 合計 prob を cap 超過時クリップ → past-self へ再配分
- [ ] `rebuild()` の win_ema carry-over を exploiter 種別ごと index 整合
- [ ] `exploiter_sel_rate`(実選択率)を host 側で集計

### Target Files
- `bot/pipeline/reinforce/case7/training/train_jax.py`

### Acceptance Criteria
- config で `exploiters: [baseline_jax_full, baseline_jax_case8]` を有効化できる
- exploiter 合計選択率が cap を超えない(単体テストで検証)

---

## Step 3: train 内 in-JAX eval + best 選択
**Target**: training(train_jax.py) / **Dependencies**: Step 1

### Overview
毎 iter 末に固定相手と in-JAX で N 戦し `eval_win` を算出、self-play win でなく
`eval_win` で best.pt を gate。iter15 の手動 sweep を制度化。

### Work Items
- [ ] `_eval_in_jax(model, opponent_name, episodes, key) -> float` 実装(argmax 決定論、host hop 無し)
- [ ] iter loop の best gate を `eval_win >= best_eval` に置換、S3 upload も連動
- [ ] `eval` config block(`eval_opponent`/`eval_episodes`/`select_metric`)読み出し

### Target Files
- `bot/pipeline/reinforce/case7/training/train_jax.py`

### Acceptance Criteria
- metrics row に `eval_win`/`eval_opponent` が毎 iter 記録される
- best.pt が `eval_win` 最大の iter のものになる(self-play win と無関係)

---

## Step 4: metrics 拡張 + 正典 config + README
**Target**: cross-cutting / **Dependencies**: Step 2, Step 3

### Overview
metrics に新 field、`pool_default.yaml` を正典化、README に pool 構成表を集約。

### Work Items
- [ ] `_write_metrics` row に `eval_win`/`eval_opponent`/`exploiter_sel_rate`
- [ ] `configs/pool_default.yaml` 作成(03-architecture の内容)
- [ ] README に pool 構成・mode 表・eval 選択を追記、loop_iter* の散逸を整理
- [ ] 旧 loop_iter* config は `configs/_archive/` へ退避(削除はしない)

### Target Files
- `bot/pipeline/reinforce/case7/training/train_jax.py`
- `bot/pipeline/reinforce/case7/configs/pool_default.yaml`
- `bot/pipeline/reinforce/case7/README.md`

### Acceptance Criteria
- `pool_default.yaml` 単体で smoke(2 iter)が完走
- README から pool 構成が一目で追える

---

## Step 5: テスト
**Target**: tests / **Dependencies**: Step 1-4

### Work Items
- [ ] `test_case8_opponent_rollout`: case8 mode で 1 game 完走 + reward 有限
- [ ] `test_exploiter_prob_cap`: cap 超過時に past-self へ再配分されることを検証
- [ ] `test_eval_in_jax_winrate`: 既知相手(noop)で eval_win≈1.0
- [ ] `test_metrics_schema`: 新 field が row に含まれる

### Target Files
- `bot/tests/pipeline/reinforce/case7/test_pool_opponents.py`(新規)

### Acceptance Criteria
- `dev/test-bot` が green(format→lint→type→pytest)

---

## Step 6: 小規模 CPU 検証(~20min)
**Target**: cross-cutting / **Dependencies**: Step 1-5

### Overview
`pool_default.yaml` で CPU ~20min 学習 → ckpt sweep で case8 混入の効果を確認。
foreground 実行必須(memory: background hang)。

### Work Items
- [ ] horizon=500, ep=8, ~16 iter で 1 run(foreground)
- [ ] eval_win 推移 + exploiter_sel_rate を metrics から可視化
- [ ] best model を vs rl_v0 / vs case8 で外部確認
- [ ] docs/experiment/reinforce に plan/result 記録

### Acceptance Criteria
- 学習信号(eval_win 単調 or 上昇)を確認、飽和暴走(reward ±10)が無い
- case8 混入有無で eval_win を比較し採否判断

---

## Step 7: GPU 段階拡大(採用時のみ)
**Target**: infra / **Dependencies**: Step 6 で採用判定

### Overview
CPU 検証で採用と出たら RunPod で iterations 拡大。memory `project_reinforce_self_snapshot_cost`
(PFSP は rollout 重い)に従い iterations 抑制 + uptime 監視。

### Work Items
- [ ] `dev/runpod train <sha> --case case7`(配線確認、cost cap 確認)
- [ ] 中間 best.pt を iter ごと S3 upload(規約: 長時間学習は中間成果物即 upload)
- [ ] 完走後 `dev/runpod pull` → 外部 eval → 採否

### Acceptance Criteria
- RunPod run 完走、中間 weights が S3 に残る
- GPU model の vs case8/v1 勝率を CPU 版と比較

### 横断的関心(全 Step)
- **horizon=500 厳守**、**飽和相手は cap で抑制**、**model 選択は外部 eval**、
  **JAX self-play は foreground**、**GPU/submit はユーザー承認**。
