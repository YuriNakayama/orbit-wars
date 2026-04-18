# テスト戦略

## 基本方針

**ノートブック挙動一致を最優先**とし、スナップショットテスト（action 全ターン一致）と統合テスト（`env.run` 完走）を中心に据える。単体テストは `WorldState` / `geometry` / `physics` の薄いものだけ用意する。カバレッジは `pipeline/case1` を `--cov=src` 指定から外し、**スナップショット diff を品質ゲートとして代替**する。

## テストピラミッド

```
           ┌─────────────────┐
           │  Snapshot test  │   ← 最上位。seed=0 の 1v1 全ターン
           │  (1 case)       │      action を diff。ノートブック一致性
           └─────────────────┘
         ┌─────────────────────┐
         │ Integration tests   │   ← env.run 完走 / 4P FFA 完走 /
         │ (~3 cases)          │      戻り値型 / DONE 到達
         └─────────────────────┘
       ┌─────────────────────────┐
       │     Unit tests          │   ← WorldState (arrival ledger, doomed)
       │     (~5 cases)          │      geometry (segment_hits_sun 等)
       └─────────────────────────┘
```

## 単体テスト (pytest)

### 対象モジュール
- `pipeline/case1/baseline/core/geometry.py`
  - `test_dist`: 既知の 2 点距離（原点と (3,4) で 5.0 等）
  - `test_segment_hits_sun_true`: 太陽中心を通る線分 → True
  - `test_segment_hits_sun_false`: 太陽から離れた線分 → False
- `pipeline/case1/baseline/core/physics.py`
  - `test_fleet_speed_monotonic`: ship数増で速度が単調増加、1000 隻で MAX_SPEED に近接
- `pipeline/case1/baseline/core/world_state.py`
  - `test_world_state_from_observation_basic`: 最小 obs (planet 1, fleet 0) で my_planets 分類が正しい
  - `test_world_state_arrival_ledger`: fleet 1 隻で `arrivals_by_planet[target_id]` に (eta, owner, ships) が積まれる
  - `test_world_state_doomed_detection`: 防衛不能な自軍星が `doomed_planets` に入る

### フィクスチャ
- `tests/pipeline/case1/conftest.py`
  - `minimal_obs`: planet 4 個 (4 折対称)、fleet 0 個の最小盤面
  - `single_fleet_obs`: 敵 fleet が 1 隻自軍星に向かう状態
  - `doomed_obs`: 自軍星の守備力を超える敵 fleet 群が向かう状態

## 統合テスト

### 対象シナリオ
- `test_env_run_1v1_completes`: `make("orbit_wars")` + `env.run([agent, agent])` が DONE に到達し、例外を投げない。
- `test_env_run_ffa4_completes`: 4P FFA でも同様に DONE 到達。
- `test_agent_returns_valid_shape`: `agent(obs)` の戻り値が `list[list[int | float]]` で、各要素が長さ 3、`from_planet_id: int`、`num_ships: int`（float 不可）。

### 実装上の注意
- `kaggle_environments.make("orbit_wars", debug=True)` でログを有効化。
- pytest marker `@pytest.mark.slow` を付け、必要に応じて `-m "not slow"` でスキップ可能に（1 エピソード数秒〜数十秒想定）。
- `asyncio_mode="auto"` は既定だが、本テストは同期でOK。

## スナップショットテスト

### セットアップ
- `tests/pipeline/case1/snapshots/episode_seed0.json` に格納するデータ:
  ```json
  {
    "version": 1,
    "seed": 0,
    "mode": "1v1",
    "kaggle_environments_version": "1.17.x",
    "actions": [
      [ /* ターン0 の p0 action */, /* p1 action */ ],
      [ /* ターン1 p0 */, /* p1 */ ],
      ...
    ],
    "final_status": ["DONE","DONE"],
    "final_scores": [123, 87]
  }
  ```
- 生成手順: `uv run python -m pipeline.case1.evaluation.snapshot_update` （このツールは Step 10 で任意作成）。

### テスト実装
```python
@pytest.mark.slow
def test_action_snapshot(snapshot_path: Path) -> None:
    expected = json.loads(snapshot_path.read_text())
    env = make("orbit_wars", configuration={"episodeSteps": 500}, debug=False)
    # seed 固定方法: env.configuration["seed"] = 0 か、np.random.seed(0) が必要
    env.run([agent, agent])
    actual_actions = [
        [step[i].get("action", []) for i in range(len(step))]
        for step in env.steps
    ]
    assert actual_actions == expected["actions"], diff_actions(actual_actions, expected["actions"])
```

### 更新ポリシー
- スナップショットが変わる変更は PR で明示的に差分レビューを行う。
- 「意図的に挙動を変えた」場合のみ `uv run python -m pipeline.case1.evaluation.snapshot_update` を実行し commit。
- 未説明の snapshot 更新は禁止（PR テンプレに「snapshot diff 根拠」チェック欄を追加する — 将来）。

## パフォーマンステスト

- selfplay CLI 実行時に計測されるため、専用テストは用意しない。
- 代わりに、selfplay の出力 `summary.csv` に `max_turn_sec`, `p95_turn_sec` を記録し、手動確認で P95<1.0s を確認する。
- 将来 `pytest-benchmark` を入れる余地あり（本 feature の scope 外）。

## テストデータ / モック

- `kaggle_environments` 自体はモックせず、実環境を利用（軽量なので許容）。
- observation fixture は手書きの dict を使う（ノートブック snippet から抜粋）。
- 乱数制御: `random.seed(0)`、`np.random.seed(0)`、`env.configuration["seed"] = 0`（サポート要確認）。

## カバレッジ方針

- **`--cov=src`** の既定を維持し、`pipeline/case1` はカバレッジ測定対象外とする。
- 品質ゲート: 統合テスト + スナップショットテスト緑 = OK。
- 将来 `pipeline/case1` を `src/` に分割した時点で、`src/features/`, `src/policies/` 等は 80%+ カバレッジ目標に組み込む（本 feature 外）。

## CI 実行コマンド

本 feature で追加するテストを CI で実行する想定コマンド:

```bash
uv run ruff format --check pipeline/case1 tests/pipeline/case1
uv run ruff check pipeline/case1 tests/pipeline/case1
uv run mypy pipeline/case1
uv run pytest tests/pipeline/case1 -v              # デフォルトは slow 含む
uv run pytest tests/pipeline/case1 -v -m "not slow" # 高速レーン
```

※ 既存の `dev/test-backend` は旧プロジェクト構造のため触らない。上記コマンドを `pipeline/case1/README.md` に転記する。

## カバレッジターゲット（最終整理）

| レイヤ | ターゲット | 計測方法 |
|--------|------------|----------|
| Unit (geometry/physics/world_state の一部) | ~5 テスト緑 | pytest 実行結果 |
| Integration (1v1 / 4P / 戻り値型) | 3 テスト緑 | pytest 実行結果 |
| Snapshot (全ターン action 一致) | 1 テスト緑 | pytest 実行結果 + snapshot 更新履歴 |
| pipeline/case1 コードカバレッジ | **対象外** | — |
| ノートブック挙動一致（NFR） | snapshot 一致 = 100% | snapshot diff |
| タイムアウト率 | P95 < 1.0s（selfplay 出力で確認） | summary.csv |
