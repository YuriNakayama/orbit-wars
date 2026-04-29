# ディレクトリリファクタリング — Web Technical Research

**作成日**: 2026-04-29
**目的**: テスタビリティ・可読性・拡張性向上のため、Python ML / Kaggle プロジェクトの組織パターンを調査

---

## Topic 1: Python ML / Kaggle プロジェクトレイアウト

### 主要パターン

1. **src/ + Hydra Config** — ロジックを `src/{models, data, utils}/` に集約、設定は `conf/` で base + experiment override。
   - ✅ 単一の真実源、experiment 増加時のコスト低
   - ⚠️ Hydra composition の学習曲線
2. **Cookiecutter Data Science (CCDS)** — `notebooks/`, `src/`, `data/`, `models/`。番号付きノートブック (`1.0-jqp-initial-exploration`) で experiment 管理。
   - ✅ 業界標準・軽量
   - ⚠️ multi-variant の組織ルールはなし
3. **PyTorch Lightning + ashleve template** — Lightning module + Hydra で `configs/experiment/` を分離。
   - ✅ プロダクション設計、checkpoint/logging が分離
   - ⚠️ 学習曲線が急

### Orbit Wars 向け推薦

**Hydra base + per-case override**: 共通デフォルト (preprocess, eval) は `conf/base.yaml`、各 case は `conf/experiments/<case>.yaml` で override。リサーチコードと提出コードを分離可能。

### Sources

- [PyTorch Lightning + Hydra template](https://github.com/ashleve/lightning-hydra-template)
- [Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/)
- [Hydra Patterns: Configuring Experiments](https://hydra.cc/docs/patterns/configuring_experiments/)

---

## Topic 2: パッケージング制約下での重複削減

### 主要パターン

1. **Vendoring + Build Script** — packaging 時に `backend/src/geometry.py` → `cases/<case>/vendor/` にコピー。case 内では `sys.path.insert(0, "vendor/")` で参照。
   - ✅ Kaggle 安全、トレーサブル
   - ⚠️ 変更時の手動同期 (script 化で緩和)
2. **Template Generator** — Jinja2 / cookiecutter で case ディレクトリを template + YAML 設定から生成。
   - ✅ DRY
   - ⚠️ template ディシプリン必要
3. **Cython / 拡張 .so シム** — 共有ロジックを binary 化して同梱。
   - ✅ Kaggle で確実に動作・高速
   - ⚠️ ビルドフロー複雑化

### Orbit Wars 向け推薦

**Vendoring + DVC integrated build** — 各 case の `dvc.yaml` に "vendor" stage を追加し、開発時は symlink、提出時はコピー。共有コアと提出コードを DVC 経由で自動同期。

### Sources

- [Vendoring in Python Packaging](https://kfchou.github.io/vendoring/)
- [Python Packaging User Guide](https://packaging.python.org/)

---

## Topic 3: ML パイプラインのテスタビリティパターン

### 主要パターン

1. **Dependency Injection + Typer/Click** — train スクリプトは config dataclass を引数として受け取り、CLI は薄い orchestrator に。
   - ✅ モック容易・グローバル状態なし
   - ⚠️ 初期 boilerplate
2. **Hexagonal (Ports & Adapters)** — コアロジック (strategy, model) を I/O (filesystem, dataset) からポート経由で分離。in-memory adapter で高速ユニットテスト。
   - ✅ 高速・disk I/O 不要
   - ⚠️ 初期 boilerplate
3. **Pytest Fixtures + Synthetic Environments** — 観測, ゲーム状態, ダミーデータセットを fixture で。`@pytest.mark.parametrize` で variant 網羅。
   - ✅ 決定的・高速
   - ⚠️ fixture 維持コスト

### Orbit Wars 向け推薦

**Hexagonal for strategy + pytest fixtures for case variants**: `strategy.py` を pure function 化し I/O 分離、`conftest.py` に obs fixture 集約、unit test for strategy + integration test for full pipeline (DVC 経由)。

### Sources

- [Hexagonal Architecture (Cockburn)](https://alistair.cockburn.us/hexagonal-architecture)
- [Pytest Fixtures Documentation](https://docs.pytest.org/)

---

## Topic 4: モノリシック関数 (700+ 行) の分割

### 主要パターン

1. **Strategy Pattern** — mission/decision logic を strategy class family に抽出 (`DefenseStrategy`, `RaidStrategy`)、context が選択して delegate。
   - ✅ テスト可能・組合せ可能
   - ⚠️ class 数増加
2. **Command Pattern** — unit action を Command オブジェクト化、dispatcher が実行。
   - ✅ undo/redo 容易
   - ⚠️ 間接性
3. **Component / ECS-lite** — entity を component (movement, combat, resource) で分解、`update(dt, state)` を持つ。
   - ✅ 単独テスト可能
   - ⚠️ ECS パラダイム理解必要

### Orbit Wars 向け推薦

**Strategy + Command hybrid**: 700 行 strategy.py を `MissionSelector`, `TargetPicker` などの strategy + 純粋関数の command builder + 薄い orchestrator に分割。各 strategy を mock game state で独立テスト。

### Sources

- [Strategy Pattern (Refactoring Guru)](https://refactoring.guru/design-patterns/strategy)
- [Game Programming Patterns: Component](https://gameprogrammingpatterns.com/component.html)

---

## Topic 5: ディレクトリ命名と experiment 組織

### 主要パターン

1. **Case + Semver + Description** — `cases/rulebase_v1.2_geometry-fix/`
   - ✅ 自己説明的・leaderboard 連携可
   - ⚠️ 冗長
2. **Dated Experiments** — `experiments/2025-04-15_faster-pathfinding/`
   - ✅ タイムライン明瞭
   - ⚠️ 非セマンティック
3. **Promotion Ladder** — `sandbox/<case>/` → 提出後 `cases/<case>/` に昇格
   - ✅ graduation 明示
   - ⚠️ 昇格時の重複

### Orbit Wars 向け推薦

**Case + Semver + status tag** — `cases/rulebase_v1.3-stable/`, `cases/imitation_v2.0-rc/`。3+ iteration で安定したものは `backend/shared/` に昇格。git tag で leaderboard checkpoint 管理。

### Sources

- [Cookiecutter Data Science Notebooks](https://cookiecutter-data-science.drivendata.org/)
- [Hydra Experiment Patterns](https://hydra.cc/docs/patterns/configuring_experiments/)

---

## 統合推薦アプローチ

**Hydra + Vendoring + DVC + Hexagonal ports** を統合スタックとして採用:

1. **Hydra** で variant 管理を config-driven に
2. **Vendoring** で Kaggle self-contained 制約を満たしつつ DRY を維持
3. **DVC** で再現性確保
4. **Hexagonal ports** で strategy ロジックをテスト可能化

これにより Kaggle 制約を遵守しつつ、5000 行規模の重複を体系的に削減できる。

---

## Orbit Wars プロジェクトの制約マッピング

| 制約 | 提案パターンとの整合性 |
|------|--------------------|
| Self-contained submission (`pipeline.md`) | Vendoring パターンが直接対応 |
| sys.path + 相対 import 必須 | Vendor 配下は相対 import で参照可 |
| DVC キャッシュ共有 | Vendor stage の hash は安定 (シンボリックリンク or hardlink) |
| 既存 AGENT_REGISTRY 結合 | リファクタ後も string-based resolve は変更不要 |
| Lint 例外 (rulebase/case1) | Strategy 分割で複雑度を本質解決 → 例外撤廃可能 |

---

## 採用候補と却下理由

### 採用候補

- **Vendoring (シンボリックリンク + 提出時コピー)**: Kaggle 制約に最適
- **Strategy Pattern 分割**: 巨大関数の本質解決
- **依存注入による CLI**: テスト可能・config 集約
- **Per-domain shared lib (`backend/src/agent_core/`)**: rulebase/imitation 横断の core を集約

### 却下候補

- **Cython 拡張 .so**: Kaggle 環境構築コスト高
- **完全 Hydra 移行**: 既存 params.yaml + DVC との共存コスト
- **Promotion Ladder (sandbox/)**: 既存 case 番号体系を破壊

---

## 参考: 競技 AI OSS の組織例

- **Lux AI Season 2**: agent ロジックを `agent/` (submission) と `lux/` (shared lib, vendored) に分離
- **Halite III**: チームの上位解は `bot/{strategy, navigation, intel}` の domain split
- **Battlecode**: Java で player package + utils package、player 配下は移譲のみ

→ vendor + domain split は競技 AI 業界での既定パターン。
