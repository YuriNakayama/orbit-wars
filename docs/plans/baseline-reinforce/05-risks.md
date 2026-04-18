# リスクと依存関係

## リスクリスト

| # | リスク | 影響 | 確率 | 緩和策 |
|---|--------|------|------|--------|
| 1 | **ノートブックとの挙動不一致** — 単一 1896 行を 12 ファイルに分割する過程で、定数のtypo、import 漏れ、演算順序のズレなどで action sequence が微妙に変わる | 高 | 中 | seed=0 の 1v1 スナップショットテストで全ターン action を diff 検知。分割後すぐ snapshot を生成し、以降の編集で差分が出れば即 CI 失敗 |
| 2 | **`uv sync` が pygame/SDL.h で失敗** — `kaggle-environments` が pygame を推移依存。macOS で `SDL.h` 不在だとソースビルドで落ちる | 高 | 高 | （a）`brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf` を README に記載、（b）`kaggle` CLI は `uv tool install kaggle` で分離インストール、（c）本体 venv が失敗しても selfplay が回る経路を確保（Kaggle env は wheel 経由で入る場合あり、失敗時は別指示） |
| 3 | **`actTimeout=1s` 超過** — `projected_state` のキャッシュヒット率が低い序盤、3-source swarm の組合せ探索でピークが走る | 高 | 低 | selfplay に `time.perf_counter` を仕込み、P95>0.8s で warning、P99>1.0s で CI fail。ノートブックオリジナルは 22s / 500 ターン = 平均 44ms なので余裕は大きい |
| 4 | **Ruff/Mypy strict が通らない** — ノートブックの巨大関数は `C901` (complexity) / `PLR0912` (too-many-branches) などに引っかかる。`Any` を挟みたくなる箇所もある | 中 | 高 | `pipeline/case1/baseline/**` に per-file-ignores を設定（スコープ外の `src/` には波及させない）。`Any` は最後の手段、Union 型で narrow 可能な箇所は narrow する |
| 5 | **kaggle CLI 認証の取り扱い事故** — `~/.kaggle/kaggle.json` や `.env` を誤ってコミット | 高 | 低 | `.gitignore` に `.kaggle/`, `kaggle.json`, `.env*` を追加。コミット前 `git status` で確認する手順を README に明記 |
| 6 | **Kaggle notebook slug 変更に伴う取得失敗** — 原典 slug は `orbit-wars-2026-reinforce` だったが、現在は `lb-897-orbit-wars-2026-reinforce` に改名されている。スコア更新のたびに slug が変わる可能性 | 中 | 中 | README に `kaggle kernels list -s "orbit-wars-2026-reinforce"` で最新 slug を確認する手順を併記。kernel-metadata.json に `id_no` も保存しておく |
| 7 | **4P FFA 環境でのクラッシュ** — `crash_exploit` は 4P 専用ロジック。1v1 で誤動作すると `env.run` がエラー終了 | 中 | 低 | agent 内で `num_players` を判定し、4P 時のみ `build_crash_exploit_missions` を呼ぶ。1v1 integrated test で事前検知 |
| 8 | **スナップショットの維持コスト** — 将来ノートブックをアップデートする度に snapshot を再生成する必要があり、意図しないバグ修正で snapshot が更新されると、バグの事実が隠蔽される | 低 | 中 | snapshot 更新時は必ず PR で diff レビュー。更新コマンドは `uv run python -m pipeline.case1.evaluation.snapshot_update` に限定し、手動編集禁止 |
| 9 | **`--cov=src` のため pipeline/case1 のカバレッジが出ない** — 数値的な品質指標が不足 | 低 | 確定 | 本 feature の scope 内では対策しない。Step 12 READMEで「カバレッジ指標は snapshot + integration test で代替」と記載 |
| 10 | **print の残存** — ノートブックは `print("debug ...")` を含む可能性があり、Submission で stdout を汚すと減点される | 低 | 中 | 移植時に `print(` を全量 grep し、`logging.getLogger(__name__).debug(...)` に置換する。CI の ruff ルールに `T201`（print検出）を `baseline/` のみ有効化 |

## 外部依存

- **Kaggle Notebook API**: `kaggle kernels pull` が 404 を返すケースあり（slug変更・削除）。Kaggle API の稼働性に依存。
- **kaggle-environments >= 1.17.0**: orbit_wars 環境の挙動がバージョンアップで変わる可能性。`pyproject.toml` で上限固定も検討余地（今回は scope外）。
- **SDL2 (OS レベル)**: pygame の native 依存。macOS では `brew install sdl2`、Linux では `apt install libsdl2-dev` が必要。
- **元ノートブック著作権**: sigmaborov 氏の Apache 2.0 ライセンスに従う。原典 URL の明記必須。

## 技術的負債

- **per-file-ignores による Ruff 緩和** — `pipeline/case1/baseline/**` の関数複雑度と行長を受容。将来 `src/` へのモジュール分割（Step B）で解消する想定。
- **Python定数とYAMLの二重管理** — `baseline/core/config.py` と `configs/baseline.yaml` が並存する。現状 Python 定数が真実、YAML は参考。チューニング feature で YAML ドリブンに切り替える際に統合する。
- **snapshot の粒度** — 1エピソードのみ。将来、異なる seed / 異なる player 割当での snapshot を追加する余地あり。
- **src/ への分割が未着手** — 本 feature 完了後 `src/agents/main.py` への再エクスポート、共通ユーティリティの `src/features/`, `src/policies/`, `src/utils/` 配置は Step B として後続。

## 未解決項目（Open Items）

- [ ] `uv sync` で pygame ビルドが通るかは OS 依存。Linux / macOS 両方の README 手順を試す必要がある。
- [ ] Kaggle 提出フロー（`kaggle competitions submit`）は scope外だが、README には **将来の提出手順の置き場所**としてプレースホルダを記載するか決定要。
- [ ] `crash_exploit` の 4P 判定は observation に `num_players` が含まれない場合 `planets` の owner set size から推定する必要あり。ノートブックの判定ロジックを移植時に精査。
- [ ] snapshot の決定性確保 — `kaggle_environments.make("orbit_wars", ...)` に seed 引数があるか未確認。`env.configuration` 経由で seed 固定できるか Step 9 で検証。
