# ディレクトリリファクタリング — アーキテクチャ設計

**作成日**: 2026-04-29

---

## 全体構成図 (refactor 後)

```
backend/
├── src/
│   ├── dataset/             # 既存維持 (変更なし)
│   ├── submit/              # 既存維持 (変更なし)
│   ├── vast/                # 既存維持 (変更なし)
│   └── evaluation/          # ★新設: case 横断の評価ロジック集約
│       ├── __init__.py
│       ├── metrics.py        # 旧 imitation/case2/3/evaluation/eval_metrics.py の統合
│       ├── vs_baseline.py    # 旧 imitation/case*/evaluation/eval_vs_baseline.py 統合
│       ├── snapshot_update.py # 旧 rulebase/case*/evaluation/snapshot_update.py 統合
│       ├── cli.py            # typer CLI (case_id 引数で分岐)
│       └── __main__.py
│
├── pipeline/
│   ├── rulebase/
│   │   ├── README.md         # ★新設: case 一覧 + ステータス表
│   │   ├── case0/            # 凍結 (touch しない)
│   │   ├── case1/            # ★Strategy+Command 分割
│   │   │   ├── main.py       # 必須 (変更なし)
│   │   │   ├── README.md     # ★新設
│   │   │   ├── baseline/
│   │   │   │   ├── agent.py  # 既存
│   │   │   │   ├── strategy/ # ★新設: 旧 strategy.py(702行) 分解
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── orchestrator.py
│   │   │   │   │   ├── mission_selector.py
│   │   │   │   │   ├── target_picker.py
│   │   │   │   │   └── order_builder.py
│   │   │   │   ├── strategy_helpers.py # 既存維持
│   │   │   │   ├── core/     # 既存 (case 独立原則のため共有しない)
│   │   │   │   └── missions/
│   │   │   ├── dump/         # ★新設 (重複コード隔離; 当面は空)
│   │   │   ├── eda/          # 既存維持
│   │   │   └── evaluation/
│   │   │       └── snapshot_update.py # ★薄ラッパー (backend/src/evaluation 呼出し)
│   │   ├── case2/, case3/    # 構造変更なし、README.md のみ追加
│   │   ├── case4/            # ★Strategy+Command 分割
│   │   │   └── baseline/strategy/  # case1 と同構造
│   │   └── case5/            # ★Strategy+Command 分割
│   │       ├── baseline/strategy/  # 同上
│   │       └── dump/agent_full.py  # ★移動 (旧 baseline/agent_full.py 2455行)
│   └── imitation/
│       ├── README.md         # ★新設
│       ├── case1/
│       │   ├── README.md     # 既存
│       │   ├── policy/       # 既存維持 (case 独立)
│       │   ├── training/     # 既存維持 (case 独立)
│       │   ├── evaluation/
│       │   │   └── eval_vs_baseline.py # ★薄ラッパー
│       │   └── dump/         # ★新設
│       ├── case2/
│       │   ├── policy/       # 既存維持
│       │   ├── training/     # 既存維持
│       │   ├── evaluation/   # ★薄ラッパー
│       │   ├── configs/      # ★削除 (params.yaml に集約)
│       │   └── dump/
│       └── case3/            # case2 同様
│
├── tests/
│   ├── evaluation/           # ★新設: backend/src/evaluation の unit test
│   │   ├── test_metrics.py
│   │   ├── test_vs_baseline.py
│   │   └── test_snapshot_update.py
│   └── pipeline/
│       ├── rulebase/
│       │   ├── case1/        # ★追加: test_strategy.py, test_core_*.py
│       │   ├── case4/        # ★追加: test_strategy.py
│       │   └── case5/        # ★追加: test_strategy.py
│       └── imitation/
│           ├── case1/        # ★追加: test_geometry.py, test_decoder.py
│           ├── case2/        # ★追加: test_geometry.py, test_decoder.py
│           └── case3/        # ★追加: test_geometry.py, test_decoder.py
│
└── pyproject.toml            # ★更新: rulebase/case1 の lint 例外撤廃

params.yaml                   # ★更新: case2/3 imitation, case1〜4 rulebase の config を集約
dvc.yaml                      # ★更新: eval stage の cmd を backend/src/evaluation/cli に変更
.claude/rules/pipeline.md     # ★更新: case 必須ファイル規約 + dump/ 推奨
```

凡例: ★ = 新規/変更箇所、(変更なし) = 既存維持

---

## 主要モジュール設計

### A. `backend/src/evaluation/` 共通評価ロジック

#### A-1. `metrics.py` (≦ 300 行)

```python
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class MetricsConfig:
    weights_path: Path
    val_parquet: Path
    config_yaml: Path | None  # None なら params.yaml 参照
    output_json: Path
    case_id: str  # "imitation_case2", "imitation_case3" etc.
    head: str = "baseline"  # "baseline" / "phase1" / "phase2"

def compute_metrics(config: MetricsConfig) -> dict:
    """旧 case2/3/evaluation/eval_metrics.py のロジックを統合。
    F1/ECE/precision/recall/per-class breakdown を計算し dict 返却。
    head 引数で baseline / phase1 / phase2 のどの head を評価するか分岐。
    """
    ...

def write_results(metrics: dict, output: Path) -> None:
    output.write_text(json.dumps(metrics, indent=2))
```

**設計判断**:
- パスは引数注入のみ (typer Option デフォルトは無し)
- case_id は文字列で受ける (副作用は path 解決のみ)
- 純粋計算 (compute) と I/O (write) を分離 → unit test で write_results をモック可能

#### A-2. `vs_baseline.py` (≦ 250 行)

```python
@dataclass
class VsBaselineConfig:
    challenger_agent: str  # AGENT_REGISTRY key
    opponents: list[str]   # AGENT_REGISTRY keys
    n_episodes: int
    seed: int
    output_json: Path

def run_vs_baseline(config: VsBaselineConfig) -> dict:
    """旧 case*/evaluation/eval_vs_baseline.py を統合。
    backend/src/dataset.run_episodes を呼び勝率と Wilson CI を返却。
    """
    ...
```

#### A-3. `snapshot_update.py` (≦ 200 行)

```python
@dataclass
class SnapshotConfig:
    case_id: str
    snapshot_dir: Path
    n_steps: int = 10

def update_snapshot(config: SnapshotConfig) -> None:
    """旧 rulebase/case*/evaluation/snapshot_update.py を統合。
    AGENT_REGISTRY 経由で agent ロード → 固定 obs で実行 → JSON 保存。
    """
    ...
```

#### A-4. `cli.py` (typer エントリポイント)

```python
import typer
app = typer.Typer()

@app.command("metrics")
def cmd_metrics(weights: Path, val: Path, output: Path, case_id: str, head: str = "baseline") -> None:
    config = MetricsConfig(weights, val, None, output, case_id, head)
    result = compute_metrics(config)
    write_results(result, output)

@app.command("vs-baseline")
def cmd_vs_baseline(challenger: str, opponents: list[str], n: int, seed: int, output: Path) -> None:
    ...

@app.command("snapshot")
def cmd_snapshot(case_id: str, snapshot_dir: Path, n_steps: int = 10) -> None:
    ...
```

**呼び出し例**:
```bash
uv run --directory backend python -m src.evaluation metrics \
  --weights pipeline/imitation/case2/policy/weights.pt \
  --val data/mart/imitation/case2/val.parquet \
  --output pipeline/imitation/case2/evaluation/results_metrics.json \
  --case-id imitation_case2 --head baseline
```

各 case 配下の `evaluation/eval_metrics.py` などは削除し、薄ラッパーまたは DVC `cmd:` の直接書き換えで対応。

---

### B. Strategy + Command 分割 (rulebase/case1, case4, case5)

#### B-1. 既存の `plan_moves(obs, state) → action_list` を以下に分解

```
plan_moves(obs, state)
  ↓
Orchestrator.plan(obs, state)
  ├─ 1. world_model.update(obs)         # 既存 core/world_model.py を利用
  ├─ 2. context = build_context(obs, state)
  ├─ 3. missions = MissionSelector.select(context)  # 候補 mission のリスト
  ├─ 4. for m in missions:
  │      target = TargetPicker.pick(m, context)
  │      orders = OrderBuilder.build(m, target, context)
  │      action_list.extend(orders)
  └─ return action_list
```

#### B-2. ファイル別責務 (case4 を例に)

```python
# strategy/orchestrator.py (≦ 80 行)
class Orchestrator:
    def __init__(
        self,
        mission_selector: MissionSelector,
        target_picker: TargetPicker,
        order_builder: OrderBuilder,
    ): ...

    def plan(self, obs: dict, state: AgentState) -> list[Action]: ...

# strategy/mission_selector.py (≦ 200 行)
class MissionSelector:
    def select(self, context: PlanContext) -> list[Mission]: ...
    # 旧 strategy.py の if/elif の塊を mission ごとに小関数へ

# strategy/target_picker.py (≦ 150 行)
class TargetPicker:
    def pick(self, mission: Mission, context: PlanContext) -> Target: ...

# strategy/order_builder.py (≦ 100 行) — 純粋関数推奨
def build_orders(mission: Mission, target: Target, context: PlanContext) -> list[Action]: ...
```

#### B-3. テスト戦略

```python
# tests/pipeline/rulebase/case4/test_strategy.py
def test_mission_selector_chooses_snipe_when_safe(safe_obs_fixture):
    selector = MissionSelector(config=DEFAULT_CONFIG)
    missions = selector.select(safe_obs_fixture)
    assert any(m.kind == "snipe" for m in missions)

def test_target_picker_prefers_high_value(threat_context_fixture):
    picker = TargetPicker()
    target = picker.pick(SnipeMission(), threat_context_fixture)
    assert target.planet_id == 7  # 既知の高価値 planet
```

各 case の strategy/ 配下は **case 内独立** で実装する (case4/strategy と case5/strategy は別物)。

#### B-4. 旧 strategy.py の互換維持

- `strategy.py` の `plan_moves` 関数シグネチャは保持し、内部で Orchestrator を呼ぶ薄いアダプタ化
- `strategy_helpers.py` 既存ヘルパは pure 化して strategy/ 内から参照
- snapshot test 通過を required 条件とする

---

### C. params.yaml 集約

#### C-1. 構造案

```yaml
# params.yaml (refactor 後)
seed: 42

data:
  selfplay_dir: data/lake/selfplay/matches
  kaggle_dir: data/lake/kaggle_episodes/matches

imitation:
  case1:
    model:
      hidden_dim: 128
      ...
    train:
      lr: 1e-3
      batch_size: 64
      ...
  case2:
    baseline:
      model: ...
      train: ...
    phase1:
      model: ...
      train: ...
  case3:
    phase2:
      model: ...
      train: ...

rulebase:
  case1:
    snipe_threshold: 0.5
    reinforcement_distance: 200
    ...
  case4:
    fleet_consolidation_min_ships: 5
    ...
```

#### C-2. アクセスパターン

```python
# 各 case の training script
import yaml
from pathlib import Path

PARAMS = yaml.safe_load(Path("params.yaml").read_text())
CONFIG = PARAMS["imitation"]["case2"]["baseline"]
```

DVC は `params:` セクションで `imitation.case2.baseline.train.lr` のようなドット記法依存を宣言。

#### C-3. 旧 configs/ ファイル

- `pipeline/imitation/case2/configs/{il_baseline.yaml, il_phase1.yaml}` を削除
- `pipeline/imitation/case3/configs/il_phase2.yaml` を削除
- `pipeline/rulebase/case1〜4/configs/baseline.yaml` (もしあれば) を削除
- 削除前に内容を `params.yaml` にマージ

---

### D. case 規約の明文化 (`.claude/rules/pipeline.md` 更新)

#### D-1. 必須要素

```
case/<id>/
├── main.py           # ★必須: Kaggle entry-point (sys.path shim + agent import)
├── __init__.py       # ★必須
├── README.md         # ★必須: case の目的・採用戦略・成績 (publicScore 等)
└── (任意) ...
```

`main.py` は以下のテンプレートを継承:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from baseline.agent import agent  # rulebase
# from policy.agent import agent  # imitation
```

#### D-2. 推奨要素

| 要素 | 用途 | 必須? |
|------|------|------|
| `baseline/` または `policy/` | エージェント実装 | 必須 (rulebase は baseline, imitation は policy) |
| `dump/` | 重複コードの隔離先 (将来の手動同期用) | 推奨 |
| `evaluation/` | 評価薄ラッパー | 推奨 |
| `tests/` (in `backend/tests/pipeline/<kind>/<case>/`) | case-local unit test | 推奨 |
| `eda/`, `notebook/`, `configs/` | 探索・設定 | optional (configs は使用しない方針) |
| `training/` | imitation のみ | imitation 必須 |

#### D-3. 規約違反禁止事項 (anti-patterns)

- `from pipeline.rulebase.case1.baseline.* import *` (cross-case import)
- top-level `print()` / `app.command()` 即時実行
- `Path("hardcoded/path")` を関数デフォルト引数に置く
- 700 行超の関数

---

### E. README ステータス表

#### E-1. `pipeline/rulebase/README.md` テンプレート

```markdown
# Rulebase Cases

| Case | Status | 概要 | publicScore | LB 順位時 | 備考 |
|------|--------|------|-------------|----------|------|
| case0 | archive | スナイパー参考実装 | n/a | n/a | 学習用 |
| case1 | active (legacy) | baseline_v1 | LB 897 | 2026-03 | strategy 分割後の参照 |
| case2 | active | baseline_v2 (OM, lookahead, harass) | n/a | n/a | OM ablation 結果あり |
| case3 | active | baseline_v3 (rollout) | n/a | n/a | |
| case4 | **production** | baseline_v4 (fleet consolidation) | 745 | 2026-04 | Strategy 分割対象 |
| case5 | active (verification) | baseline_v5 (LB1224 port) | 600 | 2026-04 | agent_full.py を dump 化 |
```

#### E-2. `pipeline/imitation/README.md` 同様の表 (il_v1〜v3)

---

## データモデル変更

なし。既存の `MatchRecord`, `AgentSpec`, `AgentTiming` (backend/src/dataset/schema/types.py) を維持。
追加するのは `MetricsConfig`, `VsBaselineConfig`, `SnapshotConfig` (backend/src/evaluation/) の dataclass のみ。

---

## インフラ変更

なし。Terraform 設定 (`infra/`) は影響範囲外。

---

## 外部統合

なし。Kaggle / Vast.ai / DVC / S3 のインターフェイスは変更なし。

---

## 後方互換性チェックリスト

| 項目 | 影響 | 対応 |
|------|------|------|
| `dvc.yaml` 出力パス | 不変 | (保持) |
| `AGENT_REGISTRY` キー | 不変 | (保持) |
| `submit/packager.py` の `case_dir.parent.parent` | 不変 (case 階層維持) | (保持) |
| 既存 snapshot JSON | 不変 (NFR1) | snapshot test 通過 = 不変保証 |
| 既存 weights.pt | 不変 | training は触らない |
| `params.yaml` の既存キー | 階層追加 | DVC stage の `params:` 宣言を更新 |
| `backend/tests/pipeline/case*/configs/` 参照 | configs/ 削除 | テスト更新 |
| imitation/case2/configs/ 削除 | training script で `params.yaml` 直読に変更 | preprocess.py / train.py 修正 |

---

## 設計トレードオフ

### 採用: case 完全独立 + dump 管理

- ✅ シンプル、case のオーナーシップ明確
- ✅ Kaggle self-contained 制約と完全整合
- ⚠️ 重複コード約 5000 行は残存 (DRY 視点では負債)
- ⚠️ 共有 core のバグ修正は手動同期が必要

→ ユーザー方針: **重複は許容、独立性を優先**

### 採用: evaluation のみ集約

- ✅ 提出に不要なコードなので case 独立原則の例外として OK
- ✅ 重複 800 行 (eval_metrics × 2 + eval_vs_baseline × 3) を解消
- ⚠️ case ローカルでカスタマイズしたい場合は薄ラッパー経由

### 不採用: training/preprocess の集約

- 採用すれば 1200 行重複が解消するが、case 固有のデータ加工ロジックが将来絡みやすい
- ユーザー方針: **training/predict は case 独立**

### 不採用: vendoring パターン

- 採用すれば共有 core の DRY と Kaggle 制約を両立できる
- ユーザー方針: **共有を作らずシンプルに保つ**
