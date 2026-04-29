# ディレクトリリファクタリング — 実装ステップ

**作成日**: 2026-04-29
**デリバリー**: 1 個の大規模 PR (feature/refactor-directory ブランチで完結)
**実装順序**: テスト追加 → リファクタ → 重複削減 (ユーザー指定)

---

## 全体スケジュール (順序的)

```
Phase A: テスト土台 (Step 1〜3)
   ↓
Phase B: 評価集約 (Step 4〜5)
   ↓
Phase C: Strategy 分割 (Step 6〜8)
   ↓
Phase D: 設定統合 (Step 9〜10)
   ↓
Phase E: 規約・ドキュメント (Step 11〜13)
   ↓
Phase F: クリーンアップ (Step 14〜15)
```

各 phase 完了後に `dev/test-backend` で全体回帰確認。

---

## Step 1: 共有 core の case-local unit test 追加 (imitation)

**Target**: backend/tests
**Dependencies**: なし

### 概要
imitation case1/2/3 の policy/{geometry, decoder}.py に対し、各 case ローカルで unit test を新規作成 (case 独立原則のため共通化しない)。

### Work Items

- [ ] `backend/tests/pipeline/imitation/case1/test_geometry.py` 作成 (≧ 12 cases)
- [ ] `backend/tests/pipeline/imitation/case1/test_decoder.py` 作成 (≧ 8 cases)
- [ ] `backend/tests/pipeline/imitation/case2/test_geometry.py` 作成
- [ ] `backend/tests/pipeline/imitation/case2/test_decoder.py` 作成
- [ ] `backend/tests/pipeline/imitation/case3/test_geometry.py` 作成
- [ ] `backend/tests/pipeline/imitation/case3/test_decoder.py` 作成
- [ ] `pytest --cov=pipeline.imitation.case1.policy.geometry` で 80% line coverage 確認

### Target Files

- `backend/tests/pipeline/imitation/case1/test_geometry.py` (新規)
- `backend/tests/pipeline/imitation/case1/test_decoder.py` (新規)
- 同上 case2, case3

### Acceptance Criteria

- 全 6 ファイル作成、各々で `pytest backend/tests/pipeline/imitation/<case>/test_*.py` が pass
- geometry.py の `predict_planet_position`, `resolve_angle`, `wrap_angle` をカバー
- decoder.py の `decode_action` を境界値含めてカバー
- coverage 80%+

---

## Step 2: 共有 core の case-local unit test 追加 (rulebase)

**Target**: backend/tests
**Dependencies**: なし (Step 1 と並行可)

### 概要
rulebase case1/4/5 の baseline/core/{geometry, physics, world_model}.py に case ローカル unit test を追加。

### Work Items

- [ ] `backend/tests/pipeline/rulebase/case1/test_core_geometry.py` 作成
- [ ] `backend/tests/pipeline/rulebase/case1/test_core_physics.py` 作成
- [ ] `backend/tests/pipeline/rulebase/case1/test_core_world_model.py` 拡充 (既存あれば)
- [ ] `backend/tests/pipeline/rulebase/case4/test_core_*.py` 同様 3 ファイル
- [ ] `backend/tests/pipeline/rulebase/case5/test_core_*.py` (case5 は core 構成が異なる; world_helpers / physics 用)

### Acceptance Criteria

- 各 case の core モジュール unit test 作成、80%+ coverage
- ruff/mypy pass

---

## Step 3: 既存 snapshot test の baseline 確立

**Target**: backend/tests
**Dependencies**: Step 1, 2

### 概要
リファクタ前の snapshot を「不変条件」として固定するため、現状の全 snapshot を再生成して commit。NFR1 (出力不変性) の検証基準にする。

### Work Items

- [ ] 各 case の snapshot 再生成スクリプト確認 (`pipeline/rulebase/case*/evaluation/snapshot_update.py` 経由)
- [ ] 全 case で snapshot 更新 → git diff で変化なしを確認 (変化があれば現状コードに非決定性あり、要調査)
- [ ] selfplay 50 戦のベースライン勝率を記録 (vs baseline_v4 等)
- [ ] ベースライン記録を `docs/plans/refactor-directory/baseline-metrics.md` に保存

### Acceptance Criteria

- snapshot diff が clean
- baseline-metrics.md 作成

---

## Step 4: backend/src/evaluation/ サブパッケージ新設

**Target**: backend/src/evaluation
**Dependencies**: Step 3

### 概要
case 横断の評価ロジックを共通化する `backend/src/evaluation/` を新設。03-architecture.md の A 章に従う。

### Work Items

- [ ] `backend/src/evaluation/__init__.py` 公開 API 定義
- [ ] `backend/src/evaluation/metrics.py` 作成 (旧 imitation/case2/3/evaluation/eval_metrics.py 統合)
- [ ] `backend/src/evaluation/vs_baseline.py` 作成 (旧 imitation/case*/evaluation/eval_vs_baseline.py 統合)
- [ ] `backend/src/evaluation/snapshot_update.py` 作成 (旧 rulebase/case*/evaluation/snapshot_update.py 統合)
- [ ] `backend/src/evaluation/cli.py` typer エントリポイント
- [ ] `backend/src/evaluation/__main__.py` (`python -m src.evaluation` 用)
- [ ] パスは引数注入のみ (デフォルト Path() 値禁止)
- [ ] `backend/tests/evaluation/test_metrics.py` 作成 (合成データで F1 / ECE 検証)
- [ ] `backend/tests/evaluation/test_vs_baseline.py` 作成 (run_episodes をモック)
- [ ] `backend/tests/evaluation/test_snapshot_update.py` 作成

### Acceptance Criteria

- 新 evaluation 経由で旧コードと同一の metrics 値が出る (回帰なし)
- coverage 80%+
- ruff/mypy pass

---

## Step 5: 各 case evaluation を薄ラッパー化 + DVC 更新

**Target**: backend/pipeline/{rulebase,imitation}, dvc.yaml
**Dependencies**: Step 4

### 概要
各 case の `evaluation/eval_metrics.py`, `eval_vs_baseline.py`, `snapshot_update.py` を削除または薄いラッパーに置換し、DVC stage の `cmd:` を `python -m src.evaluation` に変更。

### Work Items

- [ ] `pipeline/imitation/case2/evaluation/eval_metrics.py` 削除
- [ ] `pipeline/imitation/case3/evaluation/eval_metrics.py` 削除
- [ ] `pipeline/imitation/case*/evaluation/eval_vs_baseline.py` 削除 (3 ファイル)
- [ ] `pipeline/rulebase/case*/evaluation/snapshot_update.py` 削除 (4 ファイル)
- [ ] `pipeline/rulebase/case5/evaluation/{compare_v4, compare_v1, debug_splits}.py` は **残置** (アドホック)
- [ ] `dvc.yaml` の `eval_imitation_case2`, `eval_imitation_case3` 等の `cmd:` を更新
- [ ] `dvc.yaml` の `deps:` を `backend/src/evaluation/` に追加
- [ ] `dvc repro --dry` で stage が認識されることを確認
- [ ] 旧コードと出力 JSON が一致することを確認 (Step 3 baseline と diff)

### Acceptance Criteria

- 削除合計約 1300 行
- DVC pipeline が動作 (実 repro は不要、dry で OK)
- 既存 metrics 出力が回帰なし

---

## Step 6: rulebase/case4 Strategy 分割

**Target**: backend/pipeline/rulebase/case4/baseline
**Dependencies**: Step 3 (baseline metrics)

### 概要
production case の `baseline/strategy.py` を Orchestrator + MissionSelector + TargetPicker + OrderBuilder に分割。03-architecture.md の B 章に準拠。

### Work Items

- [ ] `baseline/strategy/__init__.py` 新設
- [ ] `baseline/strategy/orchestrator.py` 作成 (≦ 80 行)
- [ ] `baseline/strategy/mission_selector.py` 作成 (≦ 200 行)
- [ ] `baseline/strategy/target_picker.py` 作成 (≦ 150 行)
- [ ] `baseline/strategy/order_builder.py` 作成 (≦ 100 行, pure 関数)
- [ ] 旧 `baseline/strategy.py` の `plan_moves()` を Orchestrator 呼出しに薄ラッパー化
- [ ] `baseline/agent.py` の import 経路維持
- [ ] `backend/tests/pipeline/rulebase/case4/test_strategy.py` 作成 (mission_selector, target_picker, order_builder の各単位)
- [ ] `pytest backend/tests/pipeline/rulebase/case4/` で全 pass
- [ ] **snapshot test pass を必須** (出力不変性)
- [ ] selfplay 50 戦で vs baseline_v3 勝率が baseline と統計的差異なし (Step 3 比較)

### Acceptance Criteria

- 関数 200 行以下、cyclomatic complexity ≦ 10
- snapshot 不変
- 50 戦勝率 ± 5pp 以内

---

## Step 7: rulebase/case1 Strategy 分割

**Target**: backend/pipeline/rulebase/case1/baseline
**Dependencies**: Step 6 のパターン確立

### 概要
レガシー case (702 行 plan_moves) を分割。case4 と同手順、ただし case 独立原則のため case4 strategy/ コピーは禁止。case1 専用に書く。

### Work Items

- [ ] `pipeline/rulebase/case1/baseline/strategy/` 新設 (case4 同構造)
- [ ] orchestrator/mission_selector/target_picker/order_builder の case1 版を実装
- [ ] 旧 `strategy.py:plan_moves()` を薄ラッパー化
- [ ] `backend/tests/pipeline/rulebase/case1/test_strategy.py` 作成
- [ ] snapshot test pass 確認

### Acceptance Criteria

- 関数 200 行以下
- snapshot 不変
- ruff complexity warning 解消

---

## Step 8: rulebase/case5 Strategy 分割 + agent_full.py の dump 移動

**Target**: backend/pipeline/rulebase/case5/baseline
**Dependencies**: Step 6 (パターン確立)

### 概要
case5 はミニマリストかつ `agent_full.py` 2455 行を抱える特殊ケース。strategy/ 分割と同時に agent_full.py を dump/ に隔離。

### Work Items

- [ ] `pipeline/rulebase/case5/baseline/strategy/` 新設
- [ ] case5 用の orchestrator/mission_selector/target_picker/order_builder 実装
- [ ] `pipeline/rulebase/case5/baseline/agent_full.py` を `pipeline/rulebase/case5/dump/agent_full.py` に移動
- [ ] AGENT_REGISTRY やテストで `agent_full` が参照されていないことを確認
- [ ] 移動後、import エラーが出ないことを確認 (`pytest backend/tests/pipeline/rulebase/case5/`)
- [ ] `backend/tests/pipeline/rulebase/case5/test_strategy.py` 作成

### Acceptance Criteria

- agent_full.py は dump/ 配下、production パスからは到達不可
- snapshot 不変
- 50 戦勝率 ± 5pp 以内

---

## Step 9: params.yaml に config 集約

**Target**: params.yaml, pipeline/{imitation, rulebase}/case*/configs/
**Dependencies**: なし (他と並行可)

### 概要
`pipeline/imitation/case2/configs/`, `pipeline/imitation/case3/configs/`, `pipeline/rulebase/case*/configs/` の YAML を `params.yaml` に階層キーで統合。

### Work Items

- [ ] `params.yaml` に `imitation.case2.{baseline, phase1}`, `imitation.case3.phase2`, `rulebase.case{1,2,3,4}` セクションを追加
- [ ] 各 case の `configs/*.yaml` の内容を該当セクションにマージ
- [ ] `pipeline/imitation/case2/training/{preprocess,train}.py` を `params.yaml` 直読に修正
- [ ] `pipeline/imitation/case3/training/*.py` 同様
- [ ] `pipeline/rulebase/case*/baseline/core/config.py` (もしあれば) を更新
- [ ] `pipeline/imitation/case2/configs/` ディレクトリを削除
- [ ] `pipeline/imitation/case3/configs/` ディレクトリを削除
- [ ] `pipeline/rulebase/case*/configs/` ディレクトリを削除 (内容 params.yaml 移行後)
- [ ] `dvc.yaml` の `params:` セクションを更新 (`imitation.case2.baseline.train.lr` など)
- [ ] テスト pass 確認

### Acceptance Criteria

- 全 case の training/eval が `params.yaml` のみを config source として動作
- per-case `configs/` ディレクトリが消滅
- DVC が新キーを認識 (`dvc params diff` で確認)

---

## Step 10: ハードコードパスの依存注入化

**Target**: backend/pipeline/imitation/case*/training/, backend/pipeline/imitation/case*/evaluation/ (残存分)
**Dependencies**: Step 9

### 概要
`Path("hardcoded/path")` を関数デフォルト引数にする pattern を撤廃し、CLI の typer Option もデフォルト無し or `params.yaml` 経由に変更。

### Work Items

- [ ] `pipeline/imitation/case*/training/preprocess.py` の path 引数を全て注入化
- [ ] `pipeline/imitation/case*/training/train.py` 同様
- [ ] (Step 5 で削除済の eval スクリプトは対象外)
- [ ] `pipeline/rulebase/case*/baseline/core/config.py` 同様 (if exists)
- [ ] テスト追加: パスを差し替えて isolation テスト可能か確認

### Acceptance Criteria

- `grep -r 'Path("pipeline/' backend/pipeline/` で hits なし
- ruff pass

---

## Step 11: case 規約の明文化 (.claude/rules/pipeline.md)

**Target**: .claude/rules/pipeline.md
**Dependencies**: Step 8 (新構造確定後)

### 概要
03-architecture.md の D 章を `.claude/rules/pipeline.md` に反映。

### Work Items

- [ ] 既存 `.claude/rules/pipeline.md` を更新
- [ ] 必須ファイル (`main.py`, `__init__.py`, `README.md`) を明記
- [ ] dump/ ディレクトリの用途と命名規則を追加
- [ ] anti-patterns セクション (cross-case import, hardcoded path, 700 行関数) を追加
- [ ] 新規 case 作成手順 (template 化はしないが、必須要素チェックリスト) を追加

### Acceptance Criteria

- 規約が文書化され、レビュー時に参照可能

---

## Step 12: case ステータス README 整備

**Target**: backend/pipeline/{rulebase, imitation}/README.md, 各 case の README.md
**Dependencies**: Step 11

### 概要
case 一覧表と個別 README を作成し、active/archive/production 状態を明示。

### Work Items

- [ ] `backend/pipeline/rulebase/README.md` 新規作成 (case0〜5 のステータス表)
- [ ] `backend/pipeline/imitation/README.md` 新規作成 (case1〜3 のステータス表)
- [ ] `backend/pipeline/rulebase/case2/README.md` 新規作成 (現状欠落)
- [ ] `backend/pipeline/rulebase/case3/README.md` 新規作成
- [ ] `backend/pipeline/rulebase/case4/README.md` 新規作成
- [ ] `backend/pipeline/rulebase/case5/README.md` 新規作成
- [ ] 各 README に: 概要, 採用戦略, publicScore, ablation 結果へのリンク

### Acceptance Criteria

- 全 case に README あり
- pipeline/{rulebase,imitation}/README.md からリンク辿れる

---

## Step 13: lint 例外の撤廃

**Target**: backend/pyproject.toml
**Dependencies**: Step 7 (case1 strategy 分割完了)

### 概要
`[tool.ruff.lint.per-file-ignores]` の rulebase/case1 例外を削除し、複雑度警告ゼロを確認。

### Work Items

- [ ] `pyproject.toml` から `"pipeline/rulebase/case1/baseline/**/*.py" = ["C901", "E501", ...]` を削除
- [ ] `dev/lint` で残存警告ゼロを確認
- [ ] 残存があれば該当箇所を関数分割

### Acceptance Criteria

- `dev/lint` exit 0
- ruff per-file-ignores が 0 件

---

## Step 14: 旧コード・空ディレクトリのクリーンアップ

**Target**: backend/pipeline 全体
**Dependencies**: Step 5, 9

### 概要
削除した evaluation スクリプトや空 configs/ の残骸を回収。dump/ ディレクトリは新設のため (空 OK)、`__init__.py` の有無を統一。

### Work Items

- [ ] 全 case の `evaluation/__init__.py` 維持確認 (空 dir 化しない)
- [ ] 全 case の `dump/__init__.py` 新設 (空 case でも構造維持)
- [ ] 削除済 configs/ が git 残骸でないことを確認
- [ ] `find backend/pipeline -type d -empty` で空 dir 検出
- [ ] `__pycache__` クリーンアップ

### Acceptance Criteria

- ディレクトリ構成図 (03-architecture.md) と一致

---

## Step 15: 統合検証 (E2E)

**Target**: 全体
**Dependencies**: Step 1〜14

### 概要
PR 提出前の最終確認。

### Work Items

- [ ] `dev/test-backend` 全 pass (format → lint → mypy → pytest)
- [ ] 全 case の snapshot test pass
- [ ] selfplay 100 戦 vs baseline_v4 を実施し勝率を baseline と比較 (Step 3)
- [ ] `uv run --directory backend python -m pipeline.imitation.case1.evaluation.replay_match` で各 case (case1, case4, case5) の 1 ターン実行時間 ≦ 100ms
- [ ] `uv run --directory backend dvc repro --dry` で全 stage 認識
- [ ] `uv run --directory backend python -m submit dry-run --case rulebase/case4` で archive build 成功
- [ ] PR description 用に before/after の行数差分を集計

### Acceptance Criteria

- CI green
- 性能回帰なし
- 全 case の出力不変性確認

---

## クロスカット観点

### コミット粒度

1 個の大規模 PR だが、内部のコミットは Step 単位で切る (15 commits 程度)。これにより:
- レビュー時の bisect が可能
- 問題があれば該当 step のみ revert 可能

### CI 戦略

各 step の commit ごとに `dev/test-backend` を実行 (pre-commit hook で format/lint、push 後に GitHub Actions で full test)。Step 6/7/8 (Strategy 分割) の後は selfplay 検証も走らせる。

### ロールバック計画

- Strategy 分割後の snapshot 破壊が判明した場合 → Step 6/7/8 単位で revert
- params.yaml 集約で training が壊れた場合 → Step 9 を revert (configs/ 復活)
- evaluation 集約で metrics が回帰した場合 → Step 5 を revert (旧 eval_metrics 復活)
- 全部失敗時は feature branch を切り直し
