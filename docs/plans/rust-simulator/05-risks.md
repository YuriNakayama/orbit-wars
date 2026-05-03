# Risks and Dependencies

## Risk List

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|------------|
| 1 | Rust↔Python parity が f64 演算誤差・コードポートミス・浮動小数点比較で一致しない | High | High | (a) parity tolerance を `relative 1e-9` にして platform 差を吸収。(b) 各物理 phase ごとに **`debug_assert!` で公式 Python と同 phase の中間 state を比較**するテストを `tests/test_phase_alignment.rs` に追加。(c) `ORBIT_WARS_BACKEND=python` で **いつでも公式に戻せる**runtime fallback を維持。(d) 公式 27 cases を Rust ネイティブ test として port し、phase ごとの green を gating。 |
| 2 | 10x 高速化が PyDict / PyList 変換オーバーヘッドで未達 | Medium | Medium | (a) **state を Rust 側に保持し PyDict diff のみ書き戻す** 設計（03-architecture.md）。(b) `Python::allow_threads` で重い計算中は GIL 解放し multiprocessing 並列を活かす。(c) 未達なら `pyo3-numpy` で fleets/planets を ndarray にして zero-copy 化する fallback strategy を持つ（次イテレーション）。(d) benchmark 結果を `benchmark_results.md` に記録、未達なら段階的最適化計画を作る。 |
| 3 | rename PR が CI / Docker / DVC の参照漏れで main を壊す | High | Medium | (a) **rename を Step 1 単独タスクとして先行マージ**し、後続作業から切り離す。(b) `git grep -n "backend"` を全カテゴリ確認 (infra terraform `backend "s3"` のみ意図的に残す)。(c) PR 前に `dev/test-bot` + `docker build infra/runtime/Dockerfile` をローカル検証。(d) DVC は `dev/dvc dag` で stage 整合性を事前確認。 |
| 4 | Rust toolchain 追加で CI 時間が大幅増 | Medium | High | (a) `actions-rust-lang/setup-rust-toolchain@v1` の cache を有効化。(b) `cargo build --release` を一度走らせ、target/ を artifact cache。(c) 初回フル build は ~3 分の見込み、cache hit 時 ~30s に収まることを確認。(d) Vast.ai は Docker 焼き込みで `cargo build` を runtime からは避ける。 |
| 5 | `kaggle_environments` のバージョンアップで spec / observation 構造が変わり、Rust simulator が壊れる | Medium | Low | (a) `kaggle-environments>=1.17.0,<1.18` のように **upper bound を入れる**。(b) parity test を nightly cron で回し、新バージョンでも green か検出。(c) 公式 vendored copy は固定 SHA で同梱しているため、参照実装は安定。 |
| 6 | maturin develop が venv に extension module を install するタイミングで、既存 `uv sync` が壊れる | Medium | Low | (a) `bot/pyproject.toml` の `dependencies` に `orbit-wars-rust` を **dev only** にする選択肢を保持。(b) `dev/setup` で `uv sync` → `maturin develop` の順を厳守。(c) CI のキャッシュキーに `simulator/rust/Cargo.lock` を含める。 |
| 7 | macOS arm64 で `cargo build` がコンパイルエラー (PyO3 / rand_chacha の依存) | Low | Low | (a) PyO3 0.22 / rand_chacha 0.3 はいずれも arm64 公式サポート。(b) ローカル検証は M-series で実施。(c) 失敗時は `pyo3 = "0.21"` への downgrade を fallback。 |
| 8 | Python facade の **import side-effect (register)** が想定外の順序で発火し、テストで diff が出る | Low | Medium | (a) `bot/src/__init__.py` に `import orbit_wars_rust` を置き、package import 時に必ず確実に実行。(b) `pytest` 起動時の conftest.py に同 import を保険として追加。(c) `ORBIT_WARS_BACKEND=python` の test では公式 Python interpreter が確実に呼ばれるか debug log で検出。 |
| 9 | benchmark が 10x 未達で **PR レビュー受入が遅れる** | Medium | Medium | (a) benchmark を **PR 受入条件にしない**（CI は warning のみ）。(b) `benchmark_results.md` に macOS / Linux 両方の実測を貼り、未達のときも次の最適化計画を併記。(c) 最低 5x は確保し、10x は次イテレーションでチューニング。 |
| 10 | proptest が random actions で公式 Python と Rust の divergence を検出（規模大） | Medium | Low | (a) proptest は **shrinking で minimal failing case を抽出**できるので、検出されたらすぐに該当 phase を debug。(b) Step 12 の e2e parity が先に検出するので、proptest は最終 polishing 段階で使う。 |

## External Dependencies

- **Kaggle/kaggle-environments** (GitHub) — Apache-2.0 ライセンスで `orbit_wars` の `interpreter` / `register` API。固定 commit SHA で vendoring。
- **PyO3 + maturin** — Rust↔Python binding の build pipeline。`uv` 互換。
- **rust toolchain (rustup)** — 開発者ローカル / CI / Vast.ai docker image でセットアップ必要。
- **GitHub Actions runners** — Linux x86_64 / macOS arm64 (将来的に追加余地あり)。
- **Vast.ai 学習ノード** — Docker image `orbit-wars/runtime:<sha>` に rust toolchain + `simulator/rust` を焼き込む必要。

## Technical Debt

- **`html_renderer` / `visualizer/` は公式 Python のまま** — Rust 側で replay 表示の高速化ニーズが出れば次イテレーションで検討。
- **Python `random` 互換放棄** — `bot/src/dataset/storage/loader.py` 等で **既存の Kaggle replay scrape (Python random で生成された initial state)** をロードする際、Rust simulator で **同じ initial state から resimulate**するときは Python 経由で `generate_planets` を呼ばせるか、replay JSON から状態を直接読む拡張が必要。Step 12 の parity test で **Python が生成した state を Rust に注入する形**を取り、resimulate 用途は当面 Python backend で行う運用とする。
- **Rust 側の logging が未整備** — 当面 `eprintln!` / `tracing` 不使用。debug 時は parity assert に頼り、structured logging は次イテレーションで `tracing` 導入を検討。
- **`bot/` rename は backwards-compat なし** — 古い `git log` の path 検索 (`git log -- backend/...`) は `--follow` を付けないと壊れるが、利用頻度低く受容。

## Open Items

- **マージ戦略**: rename を **単独 PR として先にマージ**するか、simulator/ と同 PR にまとめるか。リスク #3 を考えると **分けたほうが安全**だが、ユーザー要件「PR 1 本」と矛盾。Step 1 で確認したスコープでは「PR 1 本」が選択されているので、**1 本 PR の中で rename を最初のコミットにする** 運用に決定（マージ前に rebase で順序確認）。
- **`pyo3-numpy` の採用判断**: 初期実装は plain PyDict/PyList。10x 未達なら次イテレーションで NumPy ndarray 経由に移行 (Step 13 の benchmark 結果で判断)。
- **Vendored `simulator/python/orbit_wars_vendor` のバージョン更新運用**: 公式が更新されたとき、固定 SHA からの bump をどのタイミングでするか。Cron で `kaggle-environments` PyPI バージョンと比較する仕組みは out-of-scope だが、`docs/plans/rust-simulator/UPGRADE_PROCEDURE.md` に手順を残すか検討（やらない場合は `simulator/python/NOTICE` の SHA だけ参照）。
- **CI で macOS 用 Rust build を回すか**: 現状 CI は ubuntu-latest のみ。macOS arm64 は開発者ローカルでカバー。将来的に self-play を macOS で回すケースが増えたら GitHub Actions の `macos-14` runner を追加。
