# dvc-data-control — Test Strategy

## Testing Approach

DVC はデータ・インフラ側のツールであり、Python ロジック単体テストでカバーできる範囲は限定的。
本プランのテスト戦略は以下の 3 レイヤで構成する:

1. **Unit**: pytest で Python 側変更（preprocess/train CLI 改修、packager フック）を検証。
2. **Smoke**: DVC コマンド自体の往復動作（ローカルダミー remote）を手動スクリプトで確認。
3. **Static**: Terraform fmt / validate、YAML 形式チェック、dvc.yaml 構文。

既存の `dev/test-backend` は **DVC を要求しない前提** を維持する（fixtures に実データを要求しない）。

## Unit Tests

### Backend (pytest)

| モジュール | 追加/変更するテスト |
|-----------|---------------------|
| `pipeline/imitation/case1/training/preprocess.py` | CLI 引数から `--config` 削除の確認、`params.yaml` デフォルト読み込みのテスト |
| `pipeline/imitation/case1/training/train.py` | 同上 |
| `pipeline/imitation/case1/evaluation/eval_vs_baseline.py` | params.yaml 読み込み + eval_metrics.json 書き出しのテスト |
| `src/submit/packager.py` | `ensure_weights(case_dir)` が `.dvc` stub を検知して `dvc pull` を呼ぶ／既に実体があればスキップ、の 2 パステスト（subprocess を mock） |

### 追加テストファイル

- `backend/tests/submit/test_packager_dvc_pull.py` (新規)
  - `subprocess.run` を `monkeypatch` で mock
  - stub ファイル（数百バイトのテキスト）を用意した fixtures で pull が呼ばれることを検証
  - 実体バイナリがあれば pull が呼ばれないことを検証

### パラメータ移行テスト

- `backend/tests/pipeline/imitation/case1/test_params_migration.py` (新規、単純)
  - ルート `params.yaml` が `yaml.safe_load` で parse できる
  - 必須キー（seed, data, model, train, inference）が揃っている
  - `data.out_train` / `data.out_val` のパスが存在可能（ディレクトリが .gitignore に入っていれば OK）

## Integration Tests

今回の変更は外部 API を叩かない（実 S3 は apply 後のみ）。integration test は scope 外。

## Smoke Tests (手動スクリプト)

### `dev/dvc-smoke` （新規、任意）

```bash
#!/bin/bash
set -euo pipefail
TMP_REMOTE=$(mktemp -d)
dvc remote add -d --local smoke "$TMP_REMOTE"
dvc push
dvc pull --force
dvc remote remove --local smoke
rm -rf "$TMP_REMOTE"
echo "OK: dvc push/pull round-trip works"
```

- Step 8 の Acceptance Criteria として、このスクリプトが成功することを確認。
- 実 S3 へは apply 完了後に別作業で切替。

## Static / Format Checks

- `terraform fmt -check -recursive infra/` を CI の一部に入れる（`dev/test-backend` とは別の `dev/test-infra` を作る案もある。今回は Step 9 のドキュメントで手動実行を記載）。
- `terraform validate` は `infra/environment/dev/` で実行。
- `dvc.yaml` の構文は `dvc stage list` で検証可能。CI 組込みは scope 外。

## Test Data

- 既存 fixture (`backend/tests/pipeline/imitation/case1/snapshots/`) を流用。
- DVC 絡みの新規テストは、サイズの小さな dummy binary（100-1000 byte）で stub vs 実体の分岐を確認する程度。

## Coverage Targets

- Unit: `backend/src/submit/packager.py` と `backend/pipeline/imitation/case1/training/*.py` の **改修部分 90%+**。
- Smoke: `dvc-smoke` が 1 シナリオ通る（push/pull 往復）。
- Terraform: `fmt` / `validate` / `plan` が 0 exit（plan 時は credentials mock で OK）。

## 手動確認項目（Step 完了時のチェックリスト）

- [ ] `uv run --directory backend dvc version` が 3.55+ を返す
- [ ] `dvc dag` で 4 stage の DAG が描画される
- [ ] `git status` で `data/`, `.dvc/tmp`, `.dvc/config.local`, `backend/pipeline/imitation/case1/policy/weights.pt` が **untracked に出てこない**（ignored に入っている）
- [ ] `git status` で `dvc.yaml`, `dvc.lock`, `.dvc/config`, `.dvcignore`, `params.yaml` が **追跡候補に出る**
- [ ] `terraform plan -var-file=terraform.tfvars` が実行可能（credentials 未設定の場合 plan のみ）
- [ ] `uv run python -m submit submit imitation/case1 --dry-run` が成功
- [ ] 既存 `dev/test-backend` がグリーン
- [ ] `memory/` に残る古い再現手順 (`--config ...`) を Claude 次回セッションで更新するマーカー（README に TODO）
