# ディレクトリリファクタリング — Codebase Research

**対象**: `backend/` 配下のディレクトリ構造を、テスタビリティ・可読性・拡張性の観点でリファクタリング
**作成日**: 2026-04-29
**調査元**: Explore agent によるコード考古学

---

## 1. ディレクトリ構成マップ

### Backend ルート構造

```
backend/
├── src/                      # 共有ライブラリ (dataset/submit/vast)
├── pipeline/                 # case ディレクトリ (rulebase + imitation)
├── tests/                    # pytest スイート (src + pipeline をミラー)
├── pyproject.toml            # 依存・ruff/mypy 設定
└── [config files]
```

### Rulebase Cases (5 世代)

| Case | 行数規模 | 構成上の特徴 |
|------|---------|-------------|
| case0 | 小 | 単純スナイパー参考実装 (休眠) |
| case1 | 20 .py | `baseline/{agent.py, core/, missions/, strategy.py(702行), strategy_helpers.py}` + `eda/` + `notebook/` (Kaggle export) |
| case2 | 28 .py | case1 + `opponent_model.py`, `lookahead.py`, `movements/{evacuation, followup, rear_guard}`, `harass.py`, `swarm.py` |
| case3 | 29 .py | case2 + `rollout.py` (325 行) |
| case4 | 29 .py | case3 構成 + `test_fleet_consolidation.py` |
| case5 | 15 .py | ミニマリスト: `agent.py` + `core/` + `agent_full.py` (2455 行のフォールバック) |

### Imitation Cases (3 世代)

| Case | 行数規模 | 構成上の特徴 |
|------|---------|-------------|
| case1 | 18 .py | `policy/{agent, decoder, featurizer(202行), geometry(273行), model, templates, types}` + `training/{preprocess(428行), train(484行), dataset, losses(217行)}` + `evaluation/eval_vs_baseline.py` |
| case2 | 24 .py | case1 構成 + `featurizer_phase1.py`, `agent_phase1.py`, `featurizer.py` を 18 planet × 11 global feat に拡張 (304 行) |
| case3 | 22 .py | case2 構成 + `featurizer_phase2.py` (563 行, 時系列特徴量), `agent_phase2.py` |

---

## 2. 共有ライブラリ (`backend/src/`) インベントリ

### `dataset/` — 11 モジュール

- `__init__.py` 公開 API: `AGENT_REGISTRY`, `AgentSpec`, `MatchRecord`, `RunSpec`, `list_agents`, `resolve`, `run_episodes`, `load_replay`, `list_matches`, `scrape_kaggle`
- `selfplay/agents.py` (64 行) — **コアの結合点**: 文字列ベースのレジストリ (`"baseline_v1": "pipeline.rulebase.case1.baseline.agent:agent"`)
- `selfplay/{executor, runner, report}.py` — 多戦実行オーケストレーション
- `storage/{loader, recorder, analyze, paths}.py` — Parquet I/O
- `kaggle/{scraper, client, leaderboard, state, rate_limit, types, records}.py` — Kaggle 取得
- `schema/types.py` — `MatchRecord`, `AgentSpec`, `AgentTiming` dataclasses

### `submit/` — 7 モジュール

- `packager.py` (231 行) — **重要前提**: case 階層は `pipeline/<category>/<case>/` ぴったり 2 段階を想定 (`case_dir.parent.parent` 参照)
- `validator.py` — packager → importlib で main.py をロード → kaggle_environments で dry episode
- `auth.py`, `kaggle_api.py`, `history.py`

### `vast/` — 7 モジュール

- `auth.py`, `instance.py`, `offers.py`, `cost.py`, `run_meta.py`, `cli.py` — GPU 学習オーケストレーション

**所感**: dataset / submit / vast の責務分離は明瞭で、意図的に近接した独立サブパッケージ構成。

---

## 3. ケース間コード重複 (Critical 発見事項)

### Imitation: `geometry.py` + `decoder.py` (100% 同一)

| ファイル | 行数 | diff 結果 |
|---------|------|----------|
| `imitation/case1/policy/decoder.py` | 189 | 100% 一致 |
| `imitation/case2/policy/decoder.py` | 189 | 100% 一致 |
| `imitation/case3/policy/decoder.py` | 189 | 100% 一致 |
| `imitation/case1/policy/geometry.py` | 273 | 100% 一致 |
| `imitation/case2/policy/geometry.py` | 273 | 100% 一致 |
| `imitation/case3/policy/geometry.py` | 273 | 100% 一致 |

→ **462 行の完全重複コード**。case 間横断 import が不可なため発生。

### Imitation: `featurizer.py` 系 (60–80% 重複)

- case1: 202 行, `PLANET_FEAT_DIM=11`, `GLOBAL_FEAT_DIM=6`
- case2: 304 行, `PLANET_FEAT_DIM=18`, `GLOBAL_FEAT_DIM=11` (case1 を継承拡張)
- case3: 563 行 (`featurizer_phase2.py`, 時系列拡張)
- ヘルパー関数 (`_fleet_speed`, `_fleet_target_eta`, `_read`) はテキスト一致

### Imitation: `training/{preprocess, train, dataset, losses}.py` (約 1500 行重複)

| ファイル | case1 | case2 | case3 |
|---------|-------|-------|-------|
| `preprocess.py` | 428 | 431 | 451 |
| `train.py` | 484 | 328 | 331 |
| `losses.py` | 217 | ~240 | 255 |

- case1→case2: dual-featurizer 対応の追加
- case2→case3: ロジック実質同一、phase2 featurizer 呼び出しのみ差分

### Imitation: `evaluation/eval_metrics.py` (case2/case3 で 100% 同一ロジック)

- 441 行 × 2 = 882 行
- 差分は path 文字列のみ (`s/case2/case3/g`)

### Rulebase: `core/` モジュール (約 95% 重複)

| ファイル | case1 | case2 | case3 | case4 | case5 |
|---------|-------|-------|-------|-------|-------|
| `types.py` | 50 | 50 | 50 | 50 | 39 |
| `geometry.py` | 75 | 75 | 75 | 75 | (なし) |
| `physics.py` | 279 | 279 | 279 | 279 | (なし) |
| `world_model.py` | 669 | 707 | 707 | 707 | (なし) |

→ 推定 **約 2400 行の重複**。

### 重複サマリ

| カテゴリ | スコープ | 重複率 | 推定行数 |
|---------|---------|--------|---------|
| Imitation geometry/decoder | 全 3 case | 100% | 462 行 |
| Imitation featurizer ヘルパー | case1+case2 | ~80% | 20 行 |
| Imitation training | 全 3 case | ~85% | 約 1200 行 |
| Imitation eval_metrics | case2+case3 | 100% | 約 400 行 |
| Rulebase core | 4–5 case | ~95% | 約 2400 行 |
| Rulebase missions | 中程度 | ~60% | 推定 600 行 |
| **合計** | | | **約 5000 行** |

---

## 4. テストカバレッジマップ

### `backend/tests/pipeline/` 構成

| Case | テストファイル数 | 行数 | 備考 |
|------|---------------|------|------|
| rulebase/case1 | 2 | 浅 | `test_baseline_agent`, `test_world_model` |
| rulebase/case2 | 3 | 浅 | + `test_opponent_model` |
| rulebase/case3 | 3 | 浅 | case2 と同構成 |
| rulebase/case4 | 4 | 中 | + `test_fleet_consolidation` |
| rulebase/case5 | 4 | 中 | `test_core_physics`, `test_core_world_helpers`, `test_fleet_split` |
| imitation/case1 | 7 | 677 | 最も網羅的 |
| imitation/case2 | 5 | 約 600 | |
| imitation/case3 | 3 | 275 | |

### ギャップ

- **Rulebase**: 共通ユニットテストなし (geometry, physics, world_model は agent 経由で間接テスト)
- `strategy.py` (702 行) は直接テストなし
- `missions/`, `movements/` は単体テストなし
- **Imitation**: `decoder.py`, `geometry.py` (全 case 共通) 単体テストなし
- 評価スクリプト (`eval_metrics.py`, `eval_vs_baseline.py`) はテスト対象外

### テスト品質

- **Snapshot テスト**: 浅い (action 形式の回帰検出のみ)
- **Imitation テスト**: featurizer/model は合成データで深くテスト

---

## 5. インポートと結合

### Case エントリポイント

すべて `pipeline.md` 規約に準拠:

```python
# main.py (rulebase / imitation 共通パターン)
sys.path.insert(0, str(Path.cwd()))
from baseline.agent import agent  # 相対 import
```

### AGENT_REGISTRY 結合

`backend/src/dataset/selfplay/agents.py:16-28` — 文字列ベース動的 import:

```python
AGENT_REGISTRY: dict[str, str] = {
    "baseline_v1": "pipeline.rulebase.case1.baseline.agent:agent",
    "baseline_v2": "pipeline.rulebase.case2.baseline.agent:agent",
    ...
    "il_v1": "pipeline.imitation.case1.policy.agent:agent",
    "il_v2_phase1": "pipeline.imitation.case2.policy.agent_phase1:agent",
    "il_v3": "pipeline.imitation.case3.policy.agent_phase2:agent",
}
```

- 手動で更新が必要 (オートディスカバリなし)
- 結合は import 時ではなく `resolve()` 時のみ

### Packager の前提

`backend/src/submit/packager.py:152` — `case_dir.parent.parent` で pipeline ルート参照
→ 階層構造変更時は packager も追従修正が必要

### 評価スクリプトの sys.path 操作

`imitation/case1/evaluation/eval_vs_baseline.py:24-27`:

```python
ROOT = Path(__file__).resolve().parents[4]  # 4 段階上 (脆弱)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

→ ディレクトリ階層変更で破綻。

---

## 6. テスタビリティの障害 (Critical)

### ハードコードパス

`imitation/case2/evaluation/eval_metrics.py:389-410`:

```python
weights: Path = typer.Option(Path("pipeline/imitation/case2/policy/weights.pt"), ...)
val: Path = typer.Option(Path("data/mart/imitation/case2/val.parquet"), ...)
config: Path = typer.Option(Path("pipeline/imitation/case2/configs/il_baseline.yaml"), ...)
```

→ case をまたいだ流用は path 書き換え必須・isolation テスト不可。

### 巨大関数 (テスト不可)

| ファイル | 関数 | 行数 |
|---------|------|------|
| `rulebase/case1/baseline/strategy.py` | `plan_moves()` | 702 |
| `rulebase/case5/baseline/agent_full.py` | (モノリシック) | 2455 |
| `rulebase/case1/baseline/core/world_model.py` | クラス全体 | 669 |

### 依存性注入の欠如

- `eval_metrics.py` の typer Option デフォルトはハードコード → モック/差し替え不可
- `train.py` は params.yaml を直読 → ファイル不在で import 失敗

### Lint 例外で誤魔化されている複雑度

`pyproject.toml`:

```toml
[tool.ruff.lint.per-file-ignores]
"pipeline/rulebase/case1/baseline/**/*.py" = ["C901", "E501", "PLR0912", "PLR0913", "PLR0915"]
```

→ 複雑度上限を緩めることで運用、本質的な問題は未解消。

---

## 7. 設定とエントリポイント

### DVC パイプライン

```yaml
# dvc.yaml (各 case 独立)
stages:
  preprocess_imitation_case1:
    cmd: uv run --directory backend python -m pipeline.imitation.case1.training.preprocess
  train_imitation_case1: ...
  eval_imitation_case1: ...
  preprocess_imitation_case2: ...
  ...
```

→ 各 case の training パイプラインは完全独立。共有なし。

### 設定ファイルの不整合

| Case | configs/ 有無 | 設定の所在 |
|------|--------------|-----------|
| imitation/case1 | なし | リポジトリルート `params.yaml` (移行済) |
| imitation/case2 | あり | `case2/configs/il_baseline.yaml`, `il_phase1*.yaml` |
| imitation/case3 | あり | `case3/configs/il_phase2.yaml` |
| rulebase/case1–4 | あり | `caseN/configs/baseline.yaml` |

→ case1 は `params.yaml`、case2+ は per-case configs/ が併存し統一されていない。

### Dev Scripts

`dev/` 配下に 20 スクリプト。`setup`, `format`, `lint`, `test-backend`, `submit`, `dvc-setup`, `vast-*`, `replay_one_match.py`, `plot_imitation_curves.py`, etc.

---

## 8. README / ドキュメント源泉

- 全体: `README.md`, `.claude/CLAUDE.md`, `.claude/rules/{pipeline, backend, infra, security}.md`
- Case 個別 README:
  - `backend/pipeline/rulebase/case1/README.md`
  - `backend/pipeline/imitation/case1/README.md`
  - `backend/pipeline/imitation/case2/README.md`
  - `backend/pipeline/imitation/case3/README.md`
- **欠落**: rulebase case2/3/4/5 の個別 README なし。case 番号体系の説明なし。

---

## 主要発見サマリ

### 再利用可能な既存資産

1. `backend/src/dataset/selfplay/agents.py` の **文字列ベース AGENT_REGISTRY** — 動的解決によりリファクタリングの吸収層として機能
2. `backend/src/submit/packager.py` — case 内部構造を知らない (`.submitignore` 経由) ので柔軟性あり
3. `backend/src/dataset/schema/types.py` — 型定義の単一ソース化済

### 改善が必要な箇所

1. **5000 行規模の重複** — 主に rulebase/core, imitation/{policy/geometry, decoder, training}
2. **巨大関数** — strategy.py の plan_moves (702 行), agent_full.py (2455 行)
3. **ハードコードパス** — eval/training CLI の path-as-default アンチパターン
4. **設定の不統一** — params.yaml と per-case configs/ の混在
5. **テスト浅** — rulebase 系の単体テストカバレッジ低

### 制約 (絶対遵守)

- Case ディレクトリは `sys.path.insert(0, str(Path.cwd()))` + 相対 import 規約 (`.claude/rules/pipeline.md`)
- Kaggle 提出は self-contained でなければならない (cross-package import 禁止)
- `dvc.yaml` の path は安易に変更不可 (キャッシュ整合性)
