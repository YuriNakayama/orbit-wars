# ディレクトリリファクタリング — 要件定義

**作成日**: 2026-04-29

---

## 背景と目的

### 背景

Orbit Wars プロジェクトは Kaggle 提出 self-contained 制約のもと、case ベースのイテレーティブ開発を採用してきた結果、以下の負債が蓄積している:

- **約 5000 行規模のコード重複** (rulebase/core ~2400 行, imitation/training ~1200 行, geometry/decoder 462 行 100% 一致 ほか)
- **巨大関数** — `rulebase/case1/baseline/strategy.py:plan_moves()` 702 行, `rulebase/case5/baseline/agent_full.py` 2455 行 が単体テスト不可能
- **テスタビリティの欠如** — eval/training CLI のハードコードパス、依存性注入なし、`sys.path.insert` の path-as-default アンチパターン
- **設定の不統一** — case1 は `params.yaml` に集約済、case2+ は `per-case configs/` で混在
- **case 間規約の不揃い** — case1 は `eda/`+`notebook/` 持ち、case5 は `strategy.py` 無しなど構造ばらつき
- **lint 例外で誤魔化された複雑度** — `pyproject.toml` で rulebase/case1 の C901/PLR0912/PLR0915 が無効化

### 目的

テスタビリティ・可読性・拡張性の 3 軸を改善する。

| 軸 | 達成基準 |
|---|---------|
| テスタビリティ | rulebase/case1, case4, case5 の strategy ロジックが unit test 可能 (依存注入 + 純粋関数化) |
| 可読性 | 巨大関数の 200 行以下分割、ディレクトリ規約のドキュメント化 (`.claude/rules/pipeline.md` 更新) |
| 拡張性 | 新規 case 追加時に必須テンプレート (main.py + agent export) のみ準拠で動作 |

### 重要な設計原則 (ユーザー決定事項)

- **case 間完全独立**: cross-case import は禁止、共有モジュールを設けない
- **重複は許容**: 提出に必要なコードで重複が発生する場合は `case/<id>/dump/` などで個別管理 (将来の手動同期ヒント)
- **evaluation のみ共通化**: training/predict は case 固有ロジックなので独立、評価系は `backend/src/evaluation/` 集約
- **既存 case 機能凍結**: baseline_v1〜v5, il_v1〜v3 の出力 (action 系列) は不変
- **case 番号体系維持**: 連番のまま、status は README で表示

---

## ユーザーストーリー

### 優先順位順

1. **(P0) 開発者として**、共有ユーティリティ (geometry, decoder) のバグ修正が 1 か所で済む状態にしたい — case 横断の手動同期で実現 (dump 構造で記録)
2. **(P0) 開発者として**、`strategy.py` のミッション選択ロジックを fixture-based unit test で検証したい — 巨大関数解体
3. **(P0) 開発者として**、`eval_metrics.py` を 1 つに集約して case2/case3 双方から呼び出したい — `backend/src/evaluation/` 集約
4. **(P1) 開発者として**、新規 case 追加時に最小規約 (main.py + agent export) のみで動かしたい — 規約明文化
5. **(P1) 開発者として**、設定ファイルの所在を `params.yaml` に統一して見つけやすくしたい — config 移行
6. **(P2) 開発者として**、active / archive の case 状態を README で一覧したい — ステータス表
7. **(P2) 開発者として**、lint 例外を撤廃してすべての case が同じ複雑度基準で評価される状態にしたい — strategy 分割の副産物

---

## 機能要件

### F1. Evaluation 共通化

- `backend/src/evaluation/{metrics, vs_baseline, snapshot_update}.py` を新設
- 既存の `pipeline/imitation/case2/evaluation/eval_metrics.py` (441 行) と `case3/.../eval_metrics.py` (441 行) のロジックを統合
- 既存の `pipeline/imitation/case*/evaluation/eval_vs_baseline.py` (各 ~200 行) を統合
- 既存の `pipeline/rulebase/case*/evaluation/snapshot_update.py` を統合
- CLI 化し、case_id / weights / val / config のパスを引数で受ける (依存注入)
- 各 case の `evaluation/` 配下は薄いラッパー (1〜2 行で `backend/src/evaluation/...` を呼ぶ) のみ
- DVC stage の `cmd:` パスを更新

### F2. Strategy + Command 分割 (rulebase/case1, case4, case5)

- `rulebase/case1/baseline/strategy.py:plan_moves()` (702 行) を以下に分解:
  - `MissionSelector` — 局面から mission を選択 (snipe/reinforcement/crash/etc.)
  - `TargetPicker` — mission から具体的な target を決定
  - `OrderBuilder` — 純粋関数、target から最終 action list を生成
  - `Orchestrator` — 上記を呼ぶ薄いコーディネータ
- 同様の分解を rulebase/case4 (現役チャンピオン), case5 (`agent_full.py` 2455 行) に適用
- 各分解単位を `pytest` で fixture-based test 可能にする
- 既存の snapshot test は通過させ続ける (出力 action 系列の不変性を保証)
- `pyproject.toml` の `[tool.ruff.lint.per-file-ignores]` で case1 baseline に対する複雑度例外を撤廃

### F3. 共有 core テスト追加 (case 内 unit test)

- 各 case の `tests/pipeline/{rulebase,imitation}/case<N>/` に以下の unit test を追加:
  - **Imitation**: `test_geometry.py`, `test_decoder.py` (現状 0% カバレッジ)
  - **Rulebase**: `test_core_geometry.py`, `test_core_physics.py`, `test_core_world_model.py` (case4 までは未実施)
- 共通化はせず、各 case のテストを独立に書く (重複許容方針に準拠)
- fixture は各 case ローカルに置く (`tests/pipeline/.../conftest.py`)

### F4. 設定 params.yaml 集約

- `pipeline/imitation/case2/configs/{il_baseline.yaml, il_phase1.yaml}` の内容を `params.yaml` に統合
- `pipeline/imitation/case3/configs/il_phase2.yaml` 同上
- `pipeline/rulebase/case1〜4/configs/baseline.yaml` の内容も `params.yaml` に統合 (もしあれば)
- 各 case の training/eval スクリプトは `params["imitation"]["case2"]["phase1"]` のような階層パスで参照
- per-case `configs/` ディレクトリは削除 (空になる)
- DVC `params:` セクションを更新

### F5. case 規約の明文化

- `.claude/rules/pipeline.md` を更新:
  - 必須ファイル: `main.py`, `__init__.py`, `README.md`, `agent` 関数 export
  - 推奨ファイル: `dump/` (重複コードの隔離先), `tests/` (case-local unit tests)
  - optional: `configs/` (使用しない方針だが残置可), `eda/`, `notebook/`
- `pipeline/{rulebase, imitation}/README.md` を新設し、case 一覧 + ステータス表 (active / archive / WIP) を記載

### F6. 巨大ファイルの dump 化

- `rulebase/case5/baseline/agent_full.py` (2455 行) を `case5/dump/agent_full.py` に移動 (production 利用なし、参照用)
- 移動に伴うテスト・packager・AGENT_REGISTRY の影響確認

---

## 非機能要件

### NFR1. 既存出力の不変性

- 全 case の `agent` 関数の入出力 (obs → action list) を refactor 前後で完全一致させる
- 検証手段: 既存の snapshot test 100% 通過、加えて selfplay 50 戦で勝率の統計的差異がないこと

### NFR2. テストカバレッジ目標

| 対象 | 現状 | 目標 |
|------|------|------|
| backend/src/evaluation/ | (新規) | 80% line coverage |
| rulebase/case4/baseline/ | 浅い snapshot のみ | strategy/mission/target/order の各単位で unit test 追加 |
| rulebase/case5/baseline/ | 4 tests | strategy 分割後に各単位 unit test 追加 |
| imitation/case*/policy/{geometry,decoder}/ | 0% | 80% line coverage (case ごと独立に) |

### NFR3. 性能

- agent の 1 ターン実行時間: refactor 後も 100ms 以下を維持 (現状目安)
- 計測: `uv run --directory backend python -m pipeline.imitation.case1.evaluation.replay_match` で確認

### NFR4. ドキュメント

- 各 case に README.md を新設 (case2/3/4/5 rulebase は欠落中)
- `.claude/rules/pipeline.md` を更新版で commit

### NFR5. 後方互換

- `dvc.yaml` の output path は不変 (キャッシュ整合性)
- AGENT_REGISTRY の string key (`baseline_v1`, `il_v3`) は不変
- 提出済アーカイブ (`data/submissions/`) には影響なし

---

## スコープ外 (今回扱わない)

1. **Imitation training scripts (preprocess.py, train.py, dataset.py, losses.py の重複)** — case 固有ロジック扱いで独立維持。約 1200 行の重複は許容
2. **Rulebase core モジュールの共有化** — case 間独立原則のため、約 2400 行の重複は許容
3. **Vendoring パターンの導入** — 「case 完全独立」方針により採用しない
4. **Hydra 導入** — `params.yaml` 集約で十分
5. **case 命名の semver 化** — 連番 + status 表で対応
6. **新規 case6 の作成** — strategy パターン適用は既存 case1/4/5 内で実施
7. **可視化スクリプト (debug_splits, compare_v4, replay_viewer)** — アドホック扱いで集約対象外
8. **rulebase/case0** — 休眠扱い、touch しない
9. **vast/, dataset/kaggle/, submit/ サブパッケージ** — 既存責務分離が良好なため変更なし

---

## 用語集

| Term | Description |
|------|-------------|
| case | Kaggle 提出 1 個分のディレクトリ単位 (例: `pipeline/rulebase/case4/`) |
| dump | 「重複は許容」方針下で、case 内に隔離する重複コード置き場 (`case/<id>/dump/`) |
| Strategy + Command | 巨大な意思決定関数を mission selector / target picker / order builder に分解する設計パターン |
| AGENT_REGISTRY | `backend/src/dataset/selfplay/agents.py` で管理する case → agent 関数の string-based レジストリ |
| snapshot test | 固定 obs に対する agent 出力 action 系列を JSON 化して回帰検出する pytest |
| fixture-based test | 合成 obs を `pytest.fixture` で与え mission 選択ロジックを検証する unit test |
