# dvc-data-control — Risks and Dependencies

## Risk List

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|------------|
| 1 | 複数 worktree で `dvc repro` を同時実行し cache lock が競合 | 中 | 中 | 運用ルール: 1 worktree でのみ学習実行。README に明記。将来 `dvc config core.check_update false` + 明示 lock を検討 |
| 2 | `dvc gc -c` の誤実行で remote 側データ消失 | 高 | 低 | IAM policy から `s3:DeleteObject` 外す（管理者ロール別途）。S3 bucket versioning で救済可能 |
| 3 | `cache.type symlink` で weights.pt を誤編集し cache 破壊 | 中 | 中 | `dvc unprotect` を CLAUDE.md 運用手順に明記。提出フローは packager が pull 時に read-only にする |
| 4 | Kaggle 提出で `dvc pull` が失敗し空の weights.pt が tar.gz に入る | 高 | 低 | packager の `ensure_weights` で stub 検出 → エラー終了。validator dry-run で早期検知 |
| 5 | `configs/il_baseline.yaml` 削除で他の tooling (EDA notebook 等) が壊れる | 低 | 中 | grep で参照先を洗い出し、params.yaml 移行完了後に削除。EDA 側も params.yaml を参照するよう改修 |
| 6 | Terraform state を local で bootstrap した後、S3 backend へ移行し忘れる | 中 | 中 | `infra/environment/dev/README.md` に明示手順、state.tf を初期は `/local/` 運用とコメント |
| 7 | `dvc[s3]` の依存追加で `uv.lock` が肥大し `uv sync` が遅くなる | 低 | 中 | lock のサイズ増は受容。Kaggle tar.gz には DVC 同梱しない（pipeline から import しない） |
| 8 | `data/` symlink が Git 管理下で意図せず追跡される | 中 | 低 | `.gitignore` に `/data` を明記。`git rm --cached data` が必要ならローカルで実施 |
| 9 | 既存 `memory/project_*.md` 等のメモリに残る再現手順が陳腐化 | 低 | 高 | Step 9 で README と memory の該当エントリを見直し（Claude Code 再起動後に自動で整合を取る） |
| 10 | S3 egress コスト（頻繁な pull で課金） | 低 | 低 | cache で hit するため通常は発生しない。初回 pull 850 MB のみ実費 |
| 11 | AWS credentials を誤って git commit | 致命 | 低 | `.gitignore` に `infra/**/terraform.tfvars`, `.dvc/config.local`, `.env*` を明記。Step 9 のレビューで `git status` / `git secrets` 相当で確認 |
| 12 | `dvc.lock` の merge conflict 放置でデータ不整合 | 中 | 中 | PR 運用時: `dvc.lock` の conflict は必ず `dvc repro` で再生成する旨を README に記載 |

## External Dependencies

- **AWS アカウント**: bucket 作成権限のある IAM ユーザ / 管理者ロール。今回は plan までなので不要。
- **DVC 3.x (Python)**: `uv add dvc[s3]` で解決。boto3 / aiobotocore が付随。
- **AWS CLI**: `~/.aws/credentials` に `orbit-wars` プロファイルを設定（開発者個人環境で手動）。
- **Kaggle CLI**: 既存 submit フローの一部（本プランでの変更なし）。
- **Git worktree**: 既に運用中。`data/` が symlink である前提を DVC 側の cache.dir に反映。

## Technical Debt (今回導入により発生するもの)

- **Kaggle 提出時の DVC 依存**: `dvc pull` ができない環境（例: オフライン）では提出不能。将来的には重み実体を git LFS などに二重化するオプションを検討。
- **`.dvc/config.local` の手動セットアップ**: 開発者ごとに cache.dir と aws profile を書く必要があり、オンボーディング 1 ステップ増加。`dev/dvc-setup` で自動化するが、絶対パスはホスト依存。
- **Terraform state の local bootstrap**: 初回 apply 後に `backend "s3"` へ移行する 2 段階運用は、ドキュメント化しないと後続メンバーが迷う。
- **検証プロジェクト方針で後方互換を切る**: EDA notebook や過去の memory が古い手順を参照しているケースがあり、都度修正が必要。

## Open Items (決定未済)

- **IAM user vs role**: 今回は user + access key で進める（開発者単独、macOS ローカル）。将来 GPU サーバ追加時に role + OIDC に移行する想定だが、本プランでは触れない。
- **S3 bucket 命名**: `orbit-wars-dvc-<account_id>` と `<account_id>` を含める案を推奨するが、ユーザーの AWS アカウント ID を現時点で確定していないため `terraform.tfvars` で受け取る形とする。
- **`dvc exp` の採用**: 今回は `dvc repro` ベース。`dvc exp run` / `dvc exp show` を後日導入するかは未決（memory の評価結果管理と連動させる場合に再検討）。
- **`.claude/rules/dvc.md` 新設**: `dvc.yaml` / `params.yaml` / `.dvc/**` に対するルールファイル。Step 9 で必要性判断。
- **Kaggle 提出 tar.gz に DVC stub (`.dvc` ファイル) を含めるか**: packager の除外ルールを明示する必要あり（`.submitignore` に `*.dvc` 追加）。
