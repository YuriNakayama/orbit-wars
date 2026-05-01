# ディレクトリリファクタリング — テスト戦略

**作成日**: 2026-04-29

---

## テストアプローチ

リファクタリングの **NFR1 (既存出力の不変性)** を最優先とし、3 層構造で検証する:

1. **Unit Test (新規)** — 共有 core (geometry/decoder) と Strategy 分割後の各単位
2. **Snapshot Test (既存維持 + 拡充)** — agent 出力の回帰検出
3. **Integration Test (新規)** — selfplay 多戦による勝率の統計的不変性

ユーザー指定方針:
- **Strategy / Mission の fixture-based test を重点**
- **共有 core (geometry, decoder, world_model) の unit test を重点**
- training/preprocess は scope 外

---

## Unit Tests

### Backend (pytest)

#### 共有 core (case ローカルで個別作成、共通化しない)

| 対象モジュール | テストファイル | テスト数目安 | 主要観点 |
|------------|------------|----------|---------|
| `imitation/case1/policy/geometry.py` | `tests/pipeline/imitation/case1/test_geometry.py` | 12+ | predict_planet_position, resolve_angle, wrap_angle, edge cases (0°, 360°, 負角度) |
| `imitation/case1/policy/decoder.py` | `tests/pipeline/imitation/case1/test_decoder.py` | 8+ | decode_action, NO_OP 処理, 不正 index ハンドリング |
| `imitation/case2/policy/geometry.py` | `tests/pipeline/imitation/case2/test_geometry.py` | 12+ | (case1 と独立に書く) |
| `imitation/case2/policy/decoder.py` | `tests/pipeline/imitation/case2/test_decoder.py` | 8+ | 同上 |
| `imitation/case3/policy/geometry.py` | `tests/pipeline/imitation/case3/test_geometry.py` | 12+ | 同上 |
| `imitation/case3/policy/decoder.py` | `tests/pipeline/imitation/case3/test_decoder.py` | 8+ | 同上 |
| `rulebase/case1/baseline/core/geometry.py` | `tests/pipeline/rulebase/case1/test_core_geometry.py` | 10+ | distance, angle, in_range |
| `rulebase/case1/baseline/core/physics.py` | `tests/pipeline/rulebase/case1/test_core_physics.py` | 10+ | fleet velocity, mass interaction |
| `rulebase/case1/baseline/core/world_model.py` | `tests/pipeline/rulebase/case1/test_core_world_model.py` | 15+ | update, predict, ownership change |
| 同上 case4 / case5 | 各 3 ファイル | 同様 | case 独立 |

#### Strategy + Command 分割後 (case1, case4, case5)

| 対象 | テストファイル | テスト数目安 | 観点 |
|------|------------|----------|------|
| `case4/baseline/strategy/mission_selector.py` | `tests/pipeline/rulebase/case4/test_mission_selector.py` | 15+ | mission 選択ロジック、優先度、競合解消 |
| `case4/baseline/strategy/target_picker.py` | `tests/pipeline/rulebase/case4/test_target_picker.py` | 10+ | target 選定、評価関数、tie-breaking |
| `case4/baseline/strategy/order_builder.py` | `tests/pipeline/rulebase/case4/test_order_builder.py` | 10+ | pure 関数として action list 生成 |
| `case4/baseline/strategy/orchestrator.py` | `tests/pipeline/rulebase/case4/test_orchestrator.py` | 5+ | 統合 (mock の selector/picker/builder で組み合わせ) |
| 同上 case1, case5 | 各 4 ファイル | 同様 | case 独立 |

#### backend/src/evaluation (新規共通モジュール)

| 対象 | テストファイル | テスト数目安 | 観点 |
|------|------------|----------|------|
| `backend/src/evaluation/metrics.py` | `tests/evaluation/test_metrics.py` | 15+ | F1, ECE, precision/recall, per-class breakdown |
| `backend/src/evaluation/vs_baseline.py` | `tests/evaluation/test_vs_baseline.py` | 10+ | run_episodes をモック, Wilson CI, 勝率集計 |
| `backend/src/evaluation/snapshot_update.py` | `tests/evaluation/test_snapshot_update.py` | 8+ | AGENT_REGISTRY 経由ロード, JSON 出力, n_steps 制御 |
| `backend/src/evaluation/cli.py` | `tests/evaluation/test_cli.py` | 5+ | typer CLI 引数バインディング (CliRunner) |

---

## Integration Tests

### Snapshot Test (既存 + 拡充)

#### 現状 (refactor 前)

- `tests/pipeline/rulebase/case1/snapshots/`, case2, case3, case5 にある
- `tests/pipeline/imitation/case1/snapshots/` にある
- 各 snapshot は固定 obs に対する 1 ターン action 出力 JSON

#### refactor 後の追加方針

- Step 3 で「現状の snapshot を再生成しても diff なし」を確認
- Step 6/7/8 完了時に同 snapshot test を必須通過条件
- 新規 snapshot は **追加しない** (既存固定で十分)

### Selfplay 統合テスト

#### 検証スクリプト (Step 3 / Step 15)

```bash
# baseline 計測 (refactor 前)
uv run --directory backend python -m dataset selfplay run \
  --challenger baseline_v4 --opponents baseline_v3 \
  --n 50 --seed 42 \
  --output docs/plans/refactor-directory/baseline-metrics-pre.json

# refactor 後の同条件計測
uv run --directory backend python -m dataset selfplay run \
  --challenger baseline_v4 --opponents baseline_v3 \
  --n 50 --seed 42 \
  --output docs/plans/refactor-directory/baseline-metrics-post.json
```

#### 合格基準

- 勝率の差異 ≦ 5pp (Wilson 95% CI overlapping)
- AgentTiming (1 ターン平均時間) の劣化 ≦ 50%

---

## E2E Tests

### dev/test-backend (CI)

各 Step commit ごとに以下のフルパイプラインを実行:

```bash
dev/test-backend  # ruff format --check → ruff check → mypy → pytest
```

合格基準: exit 0

### DVC Pipeline 検証

```bash
uv run --directory backend dvc repro --dry  # 全 stage 認識確認
uv run --directory backend dvc params diff  # params 変更確認
```

### Submission 検証 (Kaggle build dry-run)

各 Strategy 分割対象 case で archive ビルドが成功することを確認:

```bash
uv run --directory backend python -m submit dry-run --case rulebase/case1
uv run --directory backend python -m submit dry-run --case rulebase/case4
uv run --directory backend python -m submit dry-run --case rulebase/case5
```

合格基準: tar.gz 作成成功、validator pass

---

## Test Data / Fixtures

### Synthetic Obs Fixtures

各 case の `tests/pipeline/.../conftest.py` に以下を fixture として定義:

```python
@pytest.fixture
def safe_obs():
    """脅威なし、リソース潤沢な状態 (snipe 推奨局面)."""
    return {
        "step": 100,
        "player": 0,
        "planets": [...],  # 自軍 5 惑星、敵 3 惑星、距離十分
        "fleets": [],
        "comets": [],
    }

@pytest.fixture
def threat_obs():
    """敵艦隊が接近中の状態 (reinforcement 推奨局面)."""
    ...

@pytest.fixture
def comet_active_obs():
    """彗星活性中 (NPV 評価が変わる局面)."""
    ...
```

### Mock Agent for Strategy Tests

```python
@pytest.fixture
def mock_orchestrator(mock_mission_selector, mock_target_picker, mock_order_builder):
    return Orchestrator(mock_mission_selector, mock_target_picker, mock_order_builder)
```

### Snapshot 既存ファイル

- `tests/pipeline/rulebase/case*/snapshots/*.json` — 既存維持
- 新規 snapshot は追加しない (現状で 5 ファイル / case)

---

## Coverage Targets

| 対象 | 目標 line coverage | 計測コマンド |
|------|------------------|-----------|
| `backend/src/evaluation/` | ≧ 80% | `pytest --cov=src.evaluation` |
| `pipeline/rulebase/case4/baseline/strategy/` | ≧ 80% | `pytest --cov=pipeline.rulebase.case4.baseline.strategy` |
| `pipeline/rulebase/case1/baseline/strategy/` | ≧ 70% | (legacy のため緩め) |
| `pipeline/rulebase/case5/baseline/strategy/` | ≧ 80% | |
| `pipeline/imitation/case*/policy/{geometry,decoder}/` | ≧ 80% | 各 case 個別計測 |
| `pipeline/rulebase/case{1,4,5}/baseline/core/` | ≧ 70% | 各 case 個別計測 |

### 計測除外

- `pipeline/imitation/case*/training/` (重複だが scope 外)
- `pipeline/rulebase/case{0,2,3}/baseline/strategy.py` (Strategy 分割対象外)
- `pipeline/rulebase/case5/dump/agent_full.py` (dump 扱い、参照のみ)

---

## テスト実行順序 (CI workflow)

```yaml
# .github/workflows/ci.yml に既存
1. ruff format --check
2. ruff check
3. mypy backend/
4. pytest backend/tests/ --cov  # 全テスト + coverage
5. (refactor 中の追加) pytest --maxfail=1 backend/tests/pipeline/rulebase/case4/  # 重要 case 優先 fast-fail
```

---

## 既知の課題と回避策

### 課題: snapshot test の非決定性

- `project_imitation_case1_phase3.md` メモリで記録されている既知問題: n<300 評価は信頼不可
- **対策**: refactor 前 (Step 3) に snapshot 再生成 → diff なしを確認。diff が出る場合は seed 固定や対象テスト除外を実施。

### 課題: selfplay 50 戦の variance

- `project_case2_ablation.md` 記録: 100 戦は seed variance 大
- **対策**: 同一 seed で pre/post 比較。50 戦で差異 5pp 以内なら許容、超える場合は 200 戦に拡張。

### 課題: AGENT_REGISTRY が文字列ベースで test 検出が難しい

- 名前変更時 (例: agent_full → 削除) のリンク切れを test で検出するため、`tests/dataset/test_agents.py` (既存) で全エントリの import を試行する test を維持/拡充。

---

## 完了の定義 (Definition of Done)

各 Step の完了 = 以下すべてを満たす:

- [ ] 該当 step の Work Items がすべて完了
- [ ] `dev/test-backend` exit 0
- [ ] 新規追加コードの coverage ≧ 目標値
- [ ] snapshot test pass (改変対象 case)
- [ ] selfplay 50 戦で baseline 比 ± 5pp 以内 (Strategy 分割 step のみ)
- [ ] PR description に diff サマリ更新
