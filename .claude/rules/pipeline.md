---
paths:
  - "pipeline/**"
---

# Pipeline (pipeline/<category>/case*) Submission Rules

Kaggle 提出用 `pipeline/<category>/case*/` ディレクトリを **ローカル実行と Kaggle 提出後実行の両方で** 動作させるための規約。`<category>` は現状 `rulebase/` (case0〜case2) と `imitation/` (case1) の 2 系統。case 番号は **category ごとに独立** に 1 から振る (rulebase/case1 と imitation/case1 は別物)。

## 前提: submit 基盤の仕様

`src/submit/validator.py` と `src/submit/packager.py` が課す制約:

1. `pipeline/<category>/<case>/main.py` が **必ず case ディレクトリ直下に存在** し、トップレベルで `agent(obs)` を公開すること (`importlib.util.spec_from_file_location` で直接ロードされる)。
2. packager は `case_dir` 配下の `*.py`, `*.json`, `*.yaml`, `*.pkl`, `*.pt` 等を **case_dir 起点の相対パス構造のまま** tar.gz 化する。
3. Kaggle 実行環境では tar.gz が展開された後、`main.py` が直接実行される。**`pipeline.<category>.<case>` という上位パッケージは展開先に存在しない** ため、`from pipeline.rulebase.case1.xxx import ...` のような絶対 import は ImportError になる。

## 必須パターン (案B: 相対 import + sys.path 注入)

### 1) `pipeline/<category>/<case>/main.py` はエントリポイント専用に置く

```python
# pipeline/<category>/caseN/main.py
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from baseline.agent import agent  # noqa: E402

__all__ = ["agent"]
```

- `sys.path.insert(0, str(Path.cwd()))` で **実行時 cwd** をトップに追加 → Kaggle ランタイムが tar.gz を展開した作業ディレクトリから `baseline/` をトップレベル名で import できる。
- `from baseline.agent import agent` は **ローカルでは使わない経路** (ローカルは `pipeline.rulebase.case1.baseline.agent` 等)。両方の import 解決を一つのコードで成立させる鍵。

### `__file__` を使ってはいけない (2026-04-18 判明)

Kaggle サンドボックスでは `__file__` / `Path(__file__).resolve().parent` に依存した sys.path 注入は **Validation Episode failed (`SubmissionStatus.ERROR`)** を引き起こす。以下の書き方は **禁止**:

```python
# NG: Kaggle で Validation failed になる
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

理由 (推定): Kaggle の agent loader は展開ディレクトリを cwd にして main.py を起動するが、`__file__` は相対パス / symlink / `exec()` 経由など環境依存で化け、`.resolve()` 後のパスに `baseline/` が存在しないケースがある。

**代わりに `Path.cwd()` を使う**。Kaggle は tar.gz を展開したディレクトリが cwd になる前提で実装されているため、`Path.cwd()` なら確実に展開先を指す。

この挙動差により **ローカル validator (`src/submit/validator.py`) は cwd を移動しないため `Path.cwd()` 方式の main.py は dry-run で `ModuleNotFoundError` になる**。ローカルで `--dry-run` を通したい場合は `--skip-validation` を併用するか、自前で `cd pipeline/<category>/case<N>` してから exec する。ローカルの他経路 (`src/dataset/selfplay/agents.py` 経由の `pipeline.<category>.case<N>.baseline.agent:agent` import) は main.py を経由しないので影響を受けない。

### 2) サブパッケージ内部は相対 import に統一する

```python
# pipeline/<category>/caseN/baseline/agent.py
from .core.types import Fleet, Planet           # OK
from .strategy import plan_moves                # OK
from pipeline.rulebase.caseN.baseline.core.types import Fleet, Planet  # NG (Kaggle で解決不能)

# pipeline/<category>/caseN/baseline/missions/snipe.py
from ..core.config import SNIPE_COST_TURN_WEIGHT  # OK (パッケージ 2 階層上)
from ..strategy_helpers import target_value       # OK
```

- 相対 import は **親パッケージ名に依存しない** ため、ローカルでは `pipeline.rulebase.caseN.baseline` 経由、Kaggle では `baseline` 経由、どちらでも同一コードで解決する。
- `pipeline.<category>.caseN.baseline.*` の絶対 import は **絶対に書かない**。

### 3) サブパッケージの `__init__.py` も相対 import で公開する

```python
# pipeline/<category>/caseN/baseline/__init__.py
from .agent import agent, build_world

__all__ = ["agent", "build_world"]
```

## ディレクトリレイアウトの原則

- `pipeline/<category>/` はエージェントの大分類 (`rulebase/`, `imitation/` など)。新カテゴリを追加する場合は空の `__init__.py` を置くのみで、ロジックは置かない。
- `pipeline/<category>/case<N>/main.py` は常にエントリポイントで、20 行程度の薄い wrapper に保つ。ビジネスロジックは置かない。
- 実装本体は `pipeline/<category>/case<N>/<package>/` のサブパッケージ (例: `baseline/`, `policy/`) として階層化する。階層は可読性・メンテナンス性のため維持する。
- `pipeline/<category>/case<N>/` に `evaluation/`, `configs/`, `eda/`, `notebook/` など補助ディレクトリを置いてよい。これらは `main.py` から import されない限り Kaggle に同梱されても害はないが、tar.gz サイズは抑えたい。

## ローカル側の import 経路 (不変)

以下はローカルで使い続けて良い (相対 import 化後も解決可能):

- `src/dataset/selfplay/agents.py` — `"baseline_v1": "pipeline.rulebase.case1.baseline.agent:agent"`
- `pipeline/<category>/case<N>/evaluation/*.py` — `from pipeline.<category>.case<N>.baseline import agent as baseline_agent`
- `tests/pipeline/<category>/case<N>/*.py` — `from pipeline.<category>.case<N>.baseline.xxx import ...`

これらは Kaggle に同梱されないので、ローカル専用の絶対 import で OK。

## 新規 case を追加する手順

1. 既存カテゴリ (`rulebase/`, `imitation/`) のいずれか、または新カテゴリ配下に `pipeline/<category>/case<N>/baseline/` (または戦略名ディレクトリ) を作成し、内部 import をすべて相対 import で書く。
2. `pipeline/<category>/case<N>/main.py` を上記テンプレで作成。
3. `pipeline/<category>/case<N>/baseline/__init__.py` も相対 import。
4. `src/dataset/selfplay/agents.py` の `AGENT_REGISTRY` に `"<name>": "pipeline.<category>.case<N>.baseline.agent:agent"` を追加。
5. `dev/submit <category>/case<N> --dry-run -m "..."` を実行し、validator が `main.py` をロードして `env.run([agent, "random"])` が通ることを確認。
6. `pytest tests/pipeline/<category>/case<N>` が通ることを確認してから本番提出。

## 提出アーカイブの除外 (`pipeline/.submitignore`)

packager は `pipeline/.submitignore` を読んで tar.gz から除外するパスを決定する。**全 category / 全 case 共通** で効く (pipeline ルート直下に 1 ファイルのみ)。

### 配置と書式

- 配置: `pipeline/.submitignore` (pipeline ルート直下、1 ファイルのみ)
- 書式: gitignore 互換サブセット
  - 行頭 `#` はコメント、空行は無視
  - 末尾 `/` はディレクトリ指定 (配下全ファイルを除外)
  - それ以外は `fnmatch` でパス / ファイル名にマッチ
- パスは **case_dir 相対** で評価される (例: `eda/` は `pipeline/rulebase/case1/eda/` に効く)

### 標準除外リスト (本リポジトリ)

```
# 開発ツール (本番提出に不要)
eda/
notebook/
evaluation/
training/
configs/
```

`evaluation/snapshot_update.py` のようにローカル開発専用のスクリプトが絶対 import を含むことがあり、Kaggle 側のファイル走査で ImportError を起こして Validation Episode failed の原因になる。case ディレクトリに開発用サブディレクトリを追加したら `.submitignore` にも追記する。

### 新規開発ディレクトリを追加するときの判断基準

| ディレクトリ名 | 提出物? | `.submitignore`に入れる? |
|---|---|---|
| `baseline/`, `policy_v2/` 等 (エージェント本体) | ○ | × |
| `eda/`, `notebook/` (探索用) | × | ○ |
| `evaluation/`, `training/` (開発スクリプト) | × | ○ |
| `configs/` (ローカル参照の設定) | × (定数は`core/config.py`へ) | ○ |
| モデル重み `.pt` / `.pkl` | ○ | × |

## アンチパターン

- `main.py` 内にロジックをベタ書きする → 複数戦略の共存・単体テストが困難。
- `from pipeline.<category>.caseN.xxx import ...` をサブパッケージ内部で使う → Kaggle で ImportError。
- `sys.path.insert` を main.py 以外の箇所に書く → global 副作用が散らばり追えなくなる。
- `__init__.py` に絶対 import を書く → 相対 import 化の効果を打ち消す。
- **`__file__` を使って sys.path に注入する → Kaggle で Validation Episode failed**。必ず `Path.cwd()` を使う (上記「`__file__` を使ってはいけない」節を参照)。

## 提出クォータの挙動 (2026-04-18 判明)

Kaggle Orbit Wars は 1 日 5 提出制限だが、`SubmissionStatus.ERROR` (validation 失敗) はクォータに含まれない。つまり **validation が通らない提出は再挑戦可能** なので、エラー時は Kaggle Web UI のログを確認して原因を特定し、即座に再提出してよい。

`src/submit/` はクォータのローカルチェックを行わない (Kaggle 側が消費上限に達していれば submit が失敗するだけで、ローカル側の集計タイミングズレによる誤判定を避けるため)。現在の提出数は `uv run python -m submit submissions` で確認する。

## 検証コマンド

```bash
# ローカル import 経路確認
uv run python -c "from pipeline.rulebase.case1.baseline.agent import agent; print(agent)"

# Kaggle 側 import 経路シミュレーション (main.py を直接ロードする validator と同等)
uv run python -m submit submit rulebase/case1 --dry-run -m "dry-run verification"

# 提出前チェックスイート
dev/test-backend
```
