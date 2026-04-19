# Imitation Learning Baseline (case3) — Test Strategy

## テスト方針

本 case のテストは **4 種類のレイヤー** で構成する。`dev/test-backend` (ruff + mypy + pytest) の全レーン PASS を最終ゲートとする。

- **Unit**: データ前処理・特徴抽出・モデル forward・decoder の純粋関数検証
- **Snapshot**: 固定 obs に対する `agent(obs)` の action JSON 完全一致 (推論決定性)
- **Integration**: `env.run([case3 agent, baseline_v1])` の 1 エピソード完走と action legality
- **Determinism** (slow): 同一 seed で学習 2 回走らせて val loss 完全一致

## Unit Tests (pytest)

### 1. データ前処理

**テスト対象**: `pipeline/case3/training/preprocess.py`

- `test_rating_filter_selects_top_quartile`: 合成 parquet (rating 400〜1400) から top25% filter で上位 ~25% のみが残ることを assert。
- `test_winner_extraction_1v1`: winner=0 のフレームは player=0 視点、winner=1 は player=1 視点で抽出されることを assert。
- `test_frame_expansion_multi_actions`: 1 ターンで 3 action 出した場合に 3 行に展開される (no-op は 1 行) ことを assert。
- `test_idempotent_rerun`: 2 回実行で出力 parquet が完全一致。

### 2. 特徴抽出

**テスト対象**: `pipeline/case3/policy/featurizer.py`

- `test_featurize_shape`: 合成 obs (36 planets, 10 fleets) で planet_feats shape=(36, 11), global=(6,)。
- `test_featurize_self_enemy_encoding`: obs.player=1 のとき、owner=1 惑星は is_self=1, owner=0 惑星は is_enemy=1。
- `test_featurize_no_nan_inf`: ships=0, production=0 の edge case で全特徴が finite。
- `test_featurize_fleet_aggregation`: 同一ターゲット惑星への friendly/enemy 複数 fleet が per-planet channel で合計される。

### 3. Dataset

**テスト対象**: `pipeline/case3/training/dataset.py`

- `test_dataset_len_matches_parquet`: fixture parquet (10 行) で `len(ds) == 10`。
- `test_getitem_dtypes`: `__getitem__` 戻り値の torch.dtype が期待通り (float32/int64/bool)。
- `test_collate_batch_shape`: batch_size=4 で collate 後 shape=(4, 36, 11)。

### 4. Model forward

**テスト対象**: `pipeline/case3/policy/model.py`

- `test_forward_shapes`: dummy 入力 (B=2) で from/target/ships logits の shape を assert。
- `test_mask_effect`: planet_mask=False の惑星は logit=-inf になる。
- `test_parameter_count`: `sum(p.numel() for p in model.parameters()) < 100_000`。
- `test_backward_pass`: dummy loss.backward() が例外なく完走。

### 5. Decoder

**テスト対象**: `pipeline/case3/policy/decoder.py`

- `test_decode_greedy`: 確信度の高い from/target を選んだ action list を返す。
- `test_decode_no_op_when_threshold_low`: from_prob < 閾値 の場合 action に含まれない。
- `test_decode_masks_invalid_targets`: 到達不可能な target を除外。
- `test_decode_angle_from_aim_prediction`: 出力 angle が aim_with_prediction の戻り値と一致。

## Snapshot Test (推論決定性)

**テスト対象**: `pipeline/case3/policy/agent.py`

**パターン**: `pipeline/case1` と同じ snapshot 方式を踏襲 (`pipeline/case1/evaluation/snapshot_update.py`, `tests/pipeline/case1/snapshots/` を参考)。

- `tests/pipeline/case3/snapshots/obs_seed0_turn10.json`: seed=0 で env を 10 ターン進めた時の obs。
- `tests/pipeline/case3/snapshots/action_seed0_turn10.json`: その obs に対する expected action。
- `test_agent_deterministic`: `agent(obs)` の戻り値 JSON が snapshot と完全一致することを assert。
- snapshot 再生成スクリプト: `pipeline/case3/evaluation/snapshot_update.py` (weights.pt 更新時に再生成)。

## Integration Test

**テスト対象**: `pipeline/case3/policy/agent.py` + kaggle_environments

- `test_env_run_completes`: `make("orbit_wars", configuration={"agents":2, "seed":0}).run([case3_agent, baseline_v1])` が 500 turn 以内に DONE に到達。
- `test_all_actions_legal`: 全 step の action について `from_planet_id ∈ my_planets`, `-π ≤ angle ≤ π`, `num_ships > 0`, `num_ships ≤ planet.ships` を assert。
- `test_turn_p95_under_threshold`: 1 エピソード中の agent 1 turn あたり処理時間 p95 < 0.5s (1.0s の余裕)。

## Determinism Test (slow marker)

**テスト対象**: `pipeline/case3/training/train.py`

- `test_training_reproducibility`:
  - fixture parquet (~1,000 行) に対し seed 固定で 2 epoch 学習を 2 回実行。
  - 2 回の run で最終 val_loss が完全一致 (rtol=0) することを assert。
- `@pytest.mark.slow` を付与。デフォルトの `uv run pytest tests/pipeline/case3 -v -m "not slow"` では除外。CI の full lane でのみ実行。

## テストデータ (Fixtures)

- `tests/pipeline/case3/fixtures/mini_replay.json.gz`: 10 turn だけの mini replay (1v1)。preprocess の入力検証に使う。
- `tests/pipeline/case3/fixtures/mini_train.parquet`: 100 行の合成 parquet。dataset/train の smoke test に使う。
- `tests/pipeline/case3/fixtures/synthetic_obs.json`: 36 planets + 5 fleets の合成 obs。featurizer の assert に使う。
- これらの fixtures は `tests/pipeline/case3/fixtures/generate.py` で再生成可能にする (将来 schema 変更時に備え)。

## カバレッジターゲット

- **Unit**: `pipeline/case3/policy/` と `pipeline/case3/training/` に対して line coverage **≥ 70%**。特に featurizer / model / decoder は 85%+。
- **Integration**: 1v1 の 1 エピソード完走 (1 シナリオ)。ffa4 は optional。
- **Snapshot**: 1 ターンの決定性担保 (1 snapshot)。
- **E2E**: `pipeline/case3/evaluation/eval_vs_baseline.py` を smoke test として 3 エピソードだけ回す (`-m "not slow"` で除外可能)。

## 実行コマンド

```bash
# 高速レーン (slow テスト除外)
uv run pytest tests/pipeline/case3 -v -m "not slow"

# フルレーン (determinism 含む)
uv run pytest tests/pipeline/case3 -v

# Snapshot 再生成 (weights.pt 更新後)
uv run python -m pipeline.case3.evaluation.snapshot_update

# CI 相当 (format + lint + type + test)
dev/test-backend
```

## 合格ゲート (PR マージ基準)

- [ ] `dev/test-backend` 全レーン PASS
- [ ] `tests/pipeline/case3/` の全テスト PASS (slow 含む)
- [ ] snapshot test で推論決定性が担保されている
- [ ] Step 11 の vs baseline 評価で勝率 ≥ 50%
- [ ] 推論 turn_p95 < 1.0s (`env run` の Summary で確認)
- [ ] weights.pt サイズ < 1MB
- [ ] `pipeline/case3/` から `pipeline.case[012]` への import が 0 件 (grep 確認)
