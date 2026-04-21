# case5 (LB 1224 rulebase) — リスクと依存

## リスト

| # | リスク | 影響 | 確率 | 緩和策 |
|---|------|------|------|------|
| R-01 | notebook の 480 行 `plan_moves` を分解する際にロジックの順序依存が壊れ、勝率が落ちる | 高 | 中 | snapshot test で notebook と等価な action を出すことを保証。Step 19 で 5-10 obs について完全一致を確認。一致しない場合は分解粒度を見直す |
| R-02 | deadline 制御の `time.perf_counter()` 依存により、テストが flaky になる | 中 | 中 | `core/timing.py` を `Deadline` dataclass に切り出し、`Deadline.from_config(now=...)` に時刻を注入できる構造に。テストでは monkeypatch で fake clock を使う |
| R-03 | Apache 2.0 license の attribution 漏れ | 高 (法的) | 低 | Step 01 で `LICENSE` 配置、`baseline/__init__.py` 冒頭に出典コメント、Step 17 で LICENSE ファイル存在テスト |
| R-04 | notebook 移植中に backend.md ルール (Any 禁止/print 禁止/frozen dataclass) を見落とす | 中 | 中 | 各 Step 完了時に `dev/lint` (mypy + ruff) を必ず実行。ヒアリングで「backend.md 規範を優先」を確定済 |
| R-05 | WorldModel のフィールド追加で case4 と差分が大きく、テスト fixture の使い回しが効かない | 低 | 高 | case5 専用の `tests/pipeline/rulebase/case5/` に独立 fixture を持つ。case4 の test_world_model.py をコピーして拡張 |
| R-06 | mission の score 計算で notebook と微妙に値がずれ、選好順が変わる | 中 | 中 | snapshot test で順序まで含めて検証。`apply_score_modifiers` の各 multiplier をユニットテストで個別検証 |
| R-07 | 1 ターン 1.0s 制限を超えて Kaggle の `actTimeout` を超過 → エピソードが ERROR | 高 | 低 | `SOFT_ACT_DEADLINE=0.82` の余裕を持たせる。Step 19 の自己対戦で actual turn time をプロファイリング |
| R-08 | `frozen=True` 化で notebook の mutable パターンを破壊し、score 計算が失われる | 中 | 中 | `Mission.with_score()` ヘルパーで明示的に新インスタンス生成。テストで `id()` が変わることを確認 |
| R-09 | `pipeline/.submitignore` で `evaluation/`/`configs/` が除外されない | 高 | 低 | Step 18 で archive build を dry-run 確認。.submitignore は既存設定で対応済 (case4 で実証) |
| R-10 | Kaggle 提出枠 (5/日) を超えて検証提出を繰り返してしまう | 中 | 低 | 本番提出前に CLAUDE.md ルール通りユーザー承認を得る。validation ERROR は枠消費しないので積極利用 |
| R-11 | notebook 由来の magic number 定数群 (120+) のタイポによる挙動差 | 中 | 中 | Step 02 で `core/config.py` に移植する際、notebook と diff を取って 1 個 1 個確認。ペアプロまたは tools/diff_constants.py を作成 |
| R-12 | proactive_defense の reserve 計算が他の mission と reserve を二重消費 | 中 | 中 | settle_plan で reserve を world に書き戻し、後続 mission が認識できるようにする。`test_movements_proactive.py` で確認 |

## 外部依存

- **Kaggle Orbit Wars 環境**: `kaggle-environments` パッケージ (既存依存、変更なし)
- **`romantamrazov/orbit-star-wars-lb-max-1224` notebook**: Kaggle 上の Public notebook、Apache 2.0 ライセンス。Kaggle CLI で取得済 (`/tmp/lb1224/orbit-star-wars-lb-max-1224.ipynb`)
- **case4 (baseline_v4)**: 比較対象として使用、変更しない
- **`src/dataset/selfplay`**: AGENT_REGISTRY 追加のみ
- **`src/submit`**: 既存 packager/validator で対応 (case4 と同じ構造)

## 技術的負債

- **`core/world_model.py` ~750 行**: backend.md の 800 行上限に近い。将来的に world_model から `detection.py` / `timeline.py` を切り出す余地あり (case4 と同様の課題)
- **mission builder の重複ロジック**: 各 mission の頭で source 候補絞り込み・eta 計算・safety check が重複しがち。Step 17 完了後に `mission_utils.py` への共通化を検討 (本リリースでは out-of-scope)
- **notebook 由来の 120+ マジックナンバー**: backend.md 「マジックナンバー禁止」とは別観点で、定数名と値の妥当性は検証なし。case4 evaluation と同じく ablation で個別検証する余地

## 未解決事項

| # | 項目 | 解決時期 |
|---|------|------|
| O-01 | notebook 著者 (Roman Tamrazov) への連絡は不要か？ Apache 2.0 は黙示的に許可するため不要だが、Kaggle community 慣習として discussion でリンク報告するか検討 | Step 20 提出後 |
| O-02 | `THREE_SOURCE_SWARM_ENABLED=True` のままで重い場合、deadline スキップで自然に切れるか追加スキップ条件が必要か | Step 19 のプロファイリング結果で判断 |
| O-03 | snapshot test の許容誤差 (浮動小数誤差で angle が 1e-12 ずれた場合の扱い) | Step 17 で `pytest.approx` を使うか厳密一致か決める |
| O-04 | `evaluation/compare_v2.py` の seed 範囲 (case4 は seed 0-99 の 100 戦) | case4 と統一 |
| O-05 | base.yaml の ablation 候補リスト (どの mission を on/off するか) | Step 18 で決定。最小は rescue/recapture/proactive_defense/crash_exploit/3-source の各 on/off |
