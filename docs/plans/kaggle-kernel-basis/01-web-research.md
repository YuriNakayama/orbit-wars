# kaggle-kernel-basis — Web Research

Kaggle Kernel (Notebooks) を GPU 学習基盤として使うために確認すべき公式仕様・制約・API。本ドキュメントは初期着手時の **仮置き値** を含む。実装着手前に Phase 1 完了時点で各項目を公式 docs / API 実応答に対して再検証し、差分があれば本ファイルを更新すること。

## 1. Kaggle Kernel 実行モデル

| 項目 | 値 (要検証) | 出典 |
|------|-----|------|
| 実行モード | (a) interactive notebook (b) "Save & Run All" (commit) によるバッチ実行 | <https://www.kaggle.com/docs/notebooks> |
| 1 run の wallclock 上限 | **GPU: 9h / CPU: 12h** | docs/notebooks |
| 同時 active kernel 上限 | **~5 kernel** (CPU + GPU 合算) | docs/notebooks |
| Kernel queue | active 上限超過分は queue に入る (FIFO 推定) | 未公式、要検証 |
| Internet | 設定 ON/OFF 可。OFF だと外向き HTTPS 不可 | docs/notebooks |
| Persistent storage | なし。`/kaggle/working/` は run 単位で揮発、commit 完了で output として保存 | docs/notebooks |
| Output size 上限 | **~20 GB** (要検証、kernel commit 後の保存上限) | docs/notebooks |
| 入力 data の attach | Kaggle Dataset / Competition data を `Add Data` で attach、`/kaggle/input/<slug>/` に read-only mount | docs/notebooks |

## 2. GPU spec

| 項目 | 値 |
|------|-----|
| 利用可能 GPU | T4 x2 / P100 (週単位で切替可能性、要検証) |
| accelerator enum (Kaggle API) | `gpu-t4x2`, `gpu-p100`, `gpu-v100` (公開状況による), `tpu-v3-8`, `cpu` |
| 週次 GPU quota | **30h / week** (rolling、要検証) |
| quota 表示 | <https://www.kaggle.com/settings> → "Compute usage" |
| API 経由 quota fetch | 公式 endpoint なし (要検証、`kernels_list` 結果から runtime を集計するしかない可能性) |

## 3. Kaggle Python API (`kaggle>=1.6`)

公式 CLI と同等の機能を Python から呼べる。

### 認証
- `KAGGLE_USERNAME` + `KAGGLE_KEY` env、または `~/.kaggle/kaggle.json` (`{"username": ..., "key": ...}`) を使う。
- `KaggleApi().authenticate()` で env / config を解決。

### Kernel push

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.kernels_push_cli("/path/to/kernel_dir")
# kernel_dir には kernel-metadata.json + 実体 (.ipynb or .py) が必要
```

`kernel-metadata.json` の例:
```json
{
  "id": "username/orbit-wars-case1-20260520",
  "title": "orbit-wars case1 case1 train",
  "code_file": "main.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": ["username/orbit-wars-bot"],
  "competition_sources": [],
  "kernel_sources": []
}
```

### Kernel status polling

```python
res = api.kernel_status_cli("username/orbit-wars-case1-20260520")
# res: {"status": "queued|running|complete|error", "failureMessage": "...", ...}
```

実コードでは `kernels_status` API を CLI 経由で叩く。poll interval は 30-60s 推奨 (rate limit 回避)。

### Kernel output 取得

```python
api.kernels_output_cli("username/orbit-wars-case1-20260520", path="/local/out")
# /kaggle/working/ の中身が /local/out/ にダウンロードされる
```

### Dataset CRUD

```python
api.dataset_create_new("/path/to/dataset_dir")
api.dataset_create_version("/path/to/dataset_dir", version_notes="commit=abc1234")
api.dataset_status("username/orbit-wars-bot")
# {"status": "ready|processing|error", ...}
```

`dataset-metadata.json` の例:
```json
{
  "id": "username/orbit-wars-bot",
  "title": "Orbit Wars bot source snapshot",
  "licenses": [{"name": "Apache-2.0"}],
  "subtitle": "Code snapshot for training",
  "isPrivate": true
}
```

### Kernel list (cost-report 用)

```python
kernels = api.kernels_list(user="username", page_size=50)
# 各 kernel に title, ref, lastRunTime, language, kernelType, totalVotes 等
```

run 時間は `kernels_status` の詳細レスポンスから取れない可能性があり、`/kaggle/working/run.json` (本基盤が書き出す) に runtime を記録する方式が確実。

## 4. Orbit Wars 競技の Kaggle Kernel 制約 (要検証)

| 項目 | 想定値 | 検証方法 |
|------|--------|---------|
| Internet ON 許可 | 学習目的なら ON 可 (submission notebook のみ OFF) | <https://www.kaggle.com/competitions/orbit-wars/rules> |
| GPU 利用許可 | 学習目的なら無制限 | 同上 |
| Submission notebook 制約 | 別件 (本基盤のスコープ外、`dev/submit` 経由) | 同上 |

**重要**: 本基盤は **学習用 kernel** を回すためのもので、Kaggle competition への submit kernel ではない。submit は既存 `dev/submit` が担う。本基盤は学習結果を local の `policy/weights.pt` に取り込むまでが責務。

## 5. uv の Kaggle Kernel 内利用

Kaggle Kernel の base image (Docker image kaggle/python or kaggle/python-gpu) には:
- Python 3.10 / 3.11 (kernel 設定による)
- pytorch, numpy, pandas pre-install
- pip ≧ 23
- **uv は pre-install されていない可能性が高い** (要検証)

選択肢:
1. **本基盤推奨**: `pip install -e /kaggle/input/orbit-wars-bot/` で完結。uv 経路を完全 bypass。
2. fallback: `pip install uv && uv pip install --system /kaggle/input/orbit-wars-bot/` (実験的)

`bot/pyproject.toml` は PEP 621 準拠なので pip で build できることを 06-testing で smoke 検証する。

## 6. Rust simulator (`orbit_wars_rust`) の Kaggle 上ビルド

`simulator/rust/` は PyO3 + maturin。Kaggle Kernel 内で `pip install maturin && maturin develop --release` が成功するかは未確認。**事前ビルド** が安全:

- ローカル / CI で `manylinux2014_x86_64` wheel をビルド (`maturin build --release --target x86_64-unknown-linux-gnu`)
- 生成 wheel を Kaggle Dataset (`orbit-wars-bot`) の `wheels/` ディレクトリに同梱
- Notebook cell で `pip install /kaggle/input/orbit-wars-bot/wheels/orbit_wars_rust-*.whl` を先に実行

Phase 2 / 06-testing で実行可能性を smoke 検証する。

## 7. API rate limit

- 公式 docs 上明示なし、実用上 1 req/sec 程度なら問題なしという報告が一般的。
- 本基盤は status polling を 60s 間隔、kernel push は数分に 1 回想定で十分余裕。

## 8. 検証 TODO (Phase 1 完了基準)

- [ ] `kaggle datasets create / version` の実コマンド成功確認
- [ ] `kaggle kernels push` で notebook が QUEUED → RUNNING → COMPLETE するまでの最短サイクル smoke
- [ ] `kaggle kernels status` の実 response shape を本ドキュメントに転記
- [ ] `kaggle kernels output` で `/kaggle/working/` の任意 file が pull できる確認
- [ ] uv が Kaggle Kernel base image に **無い** ことの再確認 → pip 経路で問題ない確認
- [ ] Rust simulator wheel の manylinux ビルド + Dataset 同梱で `import orbit_wars_rust` 成功
- [ ] Orbit Wars 競技の学習用 kernel に internet ON / GPU enable の制約がないことを公式 rules で確認
