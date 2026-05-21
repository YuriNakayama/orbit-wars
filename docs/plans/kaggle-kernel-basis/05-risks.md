# kaggle-kernel-basis — Risks and Dependencies

## Risk List

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|------------|
| 1 | **Kaggle Kernel に uv が pre-install されていない** (uv 経路で環境構築失敗) | High | High | (a) 本基盤は uv を **完全 bypass**、notebook cell では `pip install -e /kaggle/input/orbit-wars-bot/` のみ使用。(b) `bot/pyproject.toml` が PEP 621 準拠で pip でも build できることを Step 10 smoke で検証。(c) もし uv 必須な依存解決が必要なら fallback として `pip install uv && uv pip install --system ...` を cell B' として用意 (Step 5 で optional 実装)。 |
| 2 | **Rust simulator (`orbit_wars_rust`) の Kaggle 上 build 失敗** | High | High | (a) **事前ビルド戦略**: ローカルまたは CI で manylinux2014_x86_64 wheel をビルド (`maturin build --release --target x86_64-unknown-linux-gnu`)、生成 wheel を Kaggle Dataset の `wheels/` に同梱。(b) notebook cell B で `pip install /kaggle/input/orbit-wars-bot/wheels/*.whl` を先行実行、cell C の `pip install -e` 時に Rust ビルドを skip させる。(c) fallback として Python-only `simulator/python/` を使う case を 03-architecture で明示。 |
| 3 | **9h GPU 上限超過** (long-running case で COMPLETE せず ERROR) | High | Medium | (a) `--max-hours 8.5` を train.py に渡し、内部 timeout で safe 終了させる (将来実装 hook)。(b) imitation case1 は ~30 分実績 (memory: project_imitation_case1_phase2)、case4/case8 は計測必要。(c) 9h 直前に best.pt を必ず保存する checkpoint をデフォルト ON 推奨。 |
| 4 | **Output 20GB 上限超過** (train.log が無限増殖 / large intermediate dump) | Medium | Medium | (a) train.py の stdout を `train.log` にリダイレクト、100MB cap (cell D で `head -c` or 自前 truncate)。(b) cell F (cleanup) で `/kaggle/working/runs/<run_id>/` 18GB 超を検知し log を切り詰め。(c) intermediate ckpt は run_dir に置かない (best.pt のみ)。 |
| 5 | **同時実行 ~5 kernel 上限超過** (新規 train が queue で永久待機) | Medium | Medium | (a) `dev/kaggle-kernel train` 起動時に `kernels_list(user=...)` で active 数を fetch、4 以上で `typer.confirm`。(b) `dev/kaggle-kernel ps` でユーザが事前確認可能。(c) 上限超過で `train` が暫く返ってこない場合の help message を表示。 |
| 6 | **週次 30h GPU quota 切れ** (新規 train 起動時に no quota エラー) | Medium | Medium | (a) `cost-report` に「今週使用時間」併記、`train` 時に残量を fetch (KaggleApi に直接 API はないため、`runs/*/run.json` の rolling 7-day 集計で近似)。(b) 1h を切ったら警告。(c) quota 切れ時の actionable hint: 「Vast/RunPod に切り替えてください」。 |
| 7 | **internet OFF 競技で機能不全** (Orbit Wars 競技は学習用 kernel に internet を許可するか要確認) | Medium | Low | (a) 01-web-research で Orbit Wars 競技 rules を確認、internet ON が許可されていることを Step 10 smoke 前に確定。(b) ON 必須なら本基盤は **学習用にのみ使用**、submit kernel は対象外と明記 (既存 `dev/submit` の責務)。(c) `--no-internet` flag を将来用に予約 (現状は警告のみ)。 |
| 8 | **KAGGLE_KEY 漏洩** (notebook 本体への埋め込み、log 流出、debug 時の env ダンプ) | High | Low | (a) `KAGGLE_KEY` は `bot/.env` (gitignored) に置き、notebook 本体には絶対に埋め込まない (cell A の env 一覧から除外)。(b) `dev/kaggle-kernel` 内部での扱いは `KaggleApi().authenticate()` の自動解決経路に委ねる。(c) `.claude/rules/security.md` 準拠で commit 前 hook で .env を検知。 |
| 9 | **Dataset version quota 上限** (Kaggle 側の per-user dataset version 上限が存在する場合) | Low | Low | (a) 公式 docs に明確な上限は記載なし、要検証 (01-web-research)。(b) per-commit 自動 version up でなく、CLI に `--dataset-bump-only` (既存最新 ver を流用) を実装し、無駄 push を抑制。(c) 上限に達した場合は古い dataset を archive する手順を README に追加。 |
| 10 | **`kaggle` Python SDK のバージョン互換破壊** (RunPod の `runpod` SDK 同様、Kaggle 側 API 変更で動かない) | Medium | Low | (a) `pyproject.toml` で `kaggle>=1.6,<2.0.0` と pin。(b) e2e 成功 commit を release tag で記録。(c) SDK 更新時は Step 10 smoke を再実行。 |
| 11 | **Kaggle 側 kernel の base image (kaggle/python-gpu) の Python version mismatch** (本 repo は Python 3.13、Kaggle は 3.10/3.11) | High | High | (a) `bot/pyproject.toml` の `requires-python` を Kaggle の供給 version (3.10+) に下げるか、Kaggle が 3.13 を提供開始するまで待つ。(b) 当面の現実解: kernel-metadata.json の `language` で kernel に古い image を要求するか、本 repo を 3.10 互換に下げる (大きい)。(c) 03-architecture / 04-steps で Python version 互換性を Step 4 acceptance に含める。 |
| 12 | **Rust wheel の binary 互換性問題** (Kaggle base image の glibc / cpu microarch と合わない) | Medium | Medium | (a) manylinux2014 標準 base で build (glibc 2.17 以上要求の image なら OK)。(b) build script は CI で固定し、再現性を確保。(c) build 失敗 fallback として Python-only sim を許容する CASE を `kaggle_kernel.config` で定義。 |
| 13 | **GPU 学習結果が CPU / Vast / RunPod 学習結果と差異** (mixed precision、CUDA 非決定論) | Medium | Medium | (a) 既存方針通り `torch.use_deterministic_algorithms(False)` 許容。(b) `run.json.train_metrics.device` を残し評価フェーズで「異 provider GPU は同条件で比較不可」と明示。(c) canonical を再現する場合は CPU で再学習する逃げ道を維持。 |
| 14 | **`dev/kaggle-kernel pull` 失敗 → 成果物消失** (Kaggle output が prune される、API 一時障害) | High | Low | (a) Kaggle output は kernel 削除まで保持される (削除タイミングはユーザ操作)、本基盤は kernel を残す方針で運用。(b) `pull` 失敗時は retry 1 回、それでもダメなら user に手動 `kaggle kernels output` を案内。(c) 重要 run は **明示的に dataset 化** して長期保存する手順を README に。 |
| 15 | **複数 provider 同時 run で `policy/weights.pt` 競合** | High | Low | (a) train.py 側で三 provider env 排他 (Step 2)。(b) `promote` は `runpod_io.artifacts.run_meta.promote_to_canonical` の共有実装を使うため、provider 間で動作一貫。(c) PR 時にどの provider の run を採用したかを明記する PR template 項目を追加。 |
| 16 | **Kaggle API rate limit** (短時間に大量 push / polling で 429) | Low | Low | (a) status polling は 60s 間隔 (デフォルト)。(b) `kernels_push` は train 1 回あたり 1 回のみ。(c) rate limit 受領時は exponential backoff (Step 6 で実装)。 |
| 17 | **パッケージ命名 `kaggle_kernel` と SDK `kaggle` の混乱** | Low | Low | (a) `__init__.py` の docstring で `import kaggle as kaggle_sdk` 規約を明記。(b) コードレビュー時にチェックリスト「自パッケージ import が `from kaggle_kernel.X import Y`、SDK が `import kaggle as kaggle_sdk` であることを確認」。(c) `dev/test-bot` の `python -c "import kaggle_kernel; import kaggle"` smoke を Step 3 acceptance に含める。 |
| 18 | **三基盤 (`dev/vast` / `dev/runpod` / `dev/kaggle-kernel`) の混同** | Low | Low | (a) 各 CLI の起動メッセージに `[kaggle-kernel]` prefix。(b) `run.json` の field でどの provider か追跡可能。(c) cost-report 出力 path で provider 別 (`kaggle_kernel_cost_report_*.md` / `runpod_cost_report_*.md` / `vast_cost_report_*.md`)。 |

## External Dependencies

- **Kaggle**: API 可用性、kernel queue 速度、dataset 容量。SLA 明示なし。
- **AWS S3 (ap-northeast-1)**: 既存 (vast/runpod と共有)。99.99% 可用性。
- **GitHub.com**: 本基盤では直接利用しない (コードは Kaggle Dataset 経由で配送)。
- **PyPI**: Kaggle Kernel 内 `pip install` で外向き HTTPS 必要。internet ON 時のみ可。
- **`kaggle` Python SDK**: 公式、月次 release 程度。1.6+ 必須。
- **`maturin`**: Rust wheel build に必要 (ローカル / CI 側のみ)。
- **`nbformat`**: notebook 生成 + validate。
- **`runpod_io` package**: `notify` / `artifacts.run_meta.promote_to_canonical` / `config.cases` を共有 import。

## Technical Debt

- **`vast.run_meta` への kaggle_kernel field 追加**: 三 provider 横断 field を vast パッケージに置く構造は不自然。Phase 2 で `bot/src/cloud_common/run_meta.py` のような中立 module への切り出しを検討 (RunPod 基盤の同名 open item と統合)。
- **`runpod_io` からの import**: `notify` / `promote_to_canonical` を kaggle_kernel から import するのは provider 間の暗黙的依存。Phase 2 で `cloud_common` に切り出すか、explicit な共通モジュール `bot/src/training_provider_common/` を作る案。
- **kernel 削除タイミング**: 古い kernel が積もると Kaggle 側 UI で見通しが悪い。`dev/kaggle-kernel gc <run_id>` で過去 kernel を削除する subcommand を Phase 2 で追加検討。
- **Spot/Interruptible 非対応**: Kaggle は無料枠なので spot 概念なし、対応不要。
- **`accelerator` 自動選択**: 現状ユーザが `--accelerator` 指定、Kaggle 側 quota が低いほうへ自動切替する logic は Phase 2 で検討。

## Open Items

- **Python 3.13 vs Kaggle 3.10/3.11 mismatch**: 本 repo `pyproject.toml` の `requires-python = ">=3.13"` を Kaggle の最新 image (要検証) に合わせて緩和する必要があるか、Step 4 着手前に判定。**最大の risk**、Phase 1 完了基準に含める。
- **Rust wheel build pipeline**: `maturin build` をローカルで毎回回すか、CI に組み込むかは Step 4 で判断。GitHub Actions の `cd-build-rust-wheel.yml` (新規) を作るのが筋。
- **Orbit Wars 学習用 kernel への internet 許可確認**: 公式 rules を 01-web-research の TODO で確認、Step 10 smoke 前に確定。
- **Kaggle Dataset の slug 命名規則**: `<user>/orbit-wars-bot` で行くか `<user>/orbit-wars-bot-<branch>` で分けるか、Step 4 で確定 (推奨: 単一 slug で version で区別)。
- **`dev/kaggle-kernel cost-report` の "今週" 定義**: Kaggle quota は rolling 7-day or fixed week? 01-web-research で確認。
- **`free_gpu_hours_remaining_at_start` の取得経路**: Kaggle API では直接取れない可能性 → `runs/*/run.json` の rolling 集計で近似する。Step 7 で実装方針確定。
