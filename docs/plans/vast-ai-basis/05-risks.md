# vast-ai-basis — Risks and Dependencies

## Risk List

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|------------|
| 1 | **Vast インスタンスが destroy されず課金高騰** (self-destroy trap が機能せず、人間が気付かないまま GPU を放置 → 数十時間 × $0.13/h = 数 USD/日) | High | Medium | (a) onstart 末尾の `trap EXIT` で **成功時のみ** self-destroy。(b) `dev/vast-cost-report` を weekly cron 想定で実行可能にし、`vastai show instances` を含めて起動中インスタンスもダッシュボード化。(c) `dev/vast-train` 起動時に既存の running instance 一覧を表示する safety check を追加（Step 9 に含める）。(d) Vast 側の最大課金通知を設定するよう README に明記 |
| 2 | **AWS クレデンシャルの Vast 側漏洩** (公共テンプレート保存、env ダンプ、デバッグログ、ssh 他人アクセス) | High | Low | (a) Vast 側 `vastai create instance` の template は **public 化しない**（個人専用 instance のみ）。(b) onstart は `env >> /etc/environment` を冒頭で行うため、`set -x` で stdout に env を吐く debug は **絶対に有効化しない**（onstart には `set -euo pipefail` のみ）。(c) IAM ポリシーは `s3:DeleteObject` を含まないため、漏洩しても破壊的操作は不可。(d) 漏洩を検知したら Terraform で `terraform taint` → `apply` でキー rotate（README に手順を明記） |
| 3 | **Vast 上で `dvc push` 失敗 → 成果物消失** (S3 一時的な障害、Vast ネットワーク断、disk full、credentials 失効) | High | Low | (a) onstart の trap で **失敗時は self-destroy しない**（人間 ssh で復旧可能）。(b) `dvc push` の戻り値を bash で検査し、失敗なら `status=failed` を `run.json` に書き、 destroy せず exit 1。(c) ローカルで `vastai logs <id>` で原因確認 → 問題解決後に `vastai ssh <id>` で再 push が可能。(d) `--disk 40` で disk 余裕を確保（model weights は < 1MB、parquet は ~200MB） |
| 4 | **train.py が canonical `policy/weights.pt` を誤上書き** (ORBIT_WARS_RUN_DIR env が脱落、テスト不足、Vast onstart のミス) | High | Low | (a) train.py に防御弾として **「`ORBIT_WARS_VAST_INSTANCE_ID` env があるとき、`ORBIT_WARS_RUN_DIR` は必須」** assertion を実装。(b) Step 2 のテストで env 指定時の経路と未指定時の経路を両方カバー。(c) 万一上書きされても `dvc commit` 前なら `dvc checkout policy/weights.pt` で復旧可能（DVC 管理下のため） |
| 5 | **Vast の host 不安定（reliability 0.99 でも切断発生）** | Medium | Medium | (a) `search_offers` の `min_reliability=0.99` をデフォルト。(b) 切断したら `vastai logs` も見えなくなるが、再実行で別 host を選択。(c) 1 run < 30 分なので影響範囲が小さい |
| 6 | **`vastai` SDK のバージョン互換破壊** (Vast 側 API 変更で SDK が動かなくなる) | Medium | Low | (a) `pyproject.toml` で `vastai>=0.3.0,<1.0.0` と pin。(b) e2e test を最低 1 度本番で通したコミットを mark し、CI ではない手動 release tag で記録 |
| 7 | **GPU 学習結果が CPU 学習結果と差異** (mixed precision、numerical stability、CUDA 非決定論) | Medium | Medium | (a) `torch.use_deterministic_algorithms(False)` のまま許容。(b) `run.json.train_metrics.device` を残し、評価フェーズで「CPU run と GPU run は同条件で比較不可」と明示。(c) Phase 2 / iter9 などの canonical を再現する場合は **CPU で再学習** する逃げ道を maintain |
| 8 | **`git clone` の認証エラー** (private repo、PAT 必要) | Medium | Medium | (a) onstart は `https://github.com/<user>/orbit-wars.git` の HTTPS clone を試す。private 化されている場合は `--env '-e GIT_PAT=...'` で PAT を注入し、onstart 内で URL に埋め込む。(b) public/private 状態を実装着手時に最終確認。(c) 必要なら `dev/vast-train` で `GIT_PAT` を `~/.env` から読む |
| 9 | **`uv sync --locked` がリモート PyTorch CUDA wheel 取得で失敗** (Vast の base image が pytorch image でも、uv が再 install を試みる) | Medium | Medium | (a) base image を `pytorch/pytorch:2.6.0-cuda12.4-runtime` に固定し、torch は image の system Python に既存。(b) onstart で `uv sync --frozen` ではなく `uv sync --locked` を使い、wheel が見つからなければ詳細ログ。(c) 切り戻し: pure python 環境を捨て、image の system pytorch + pip install で運用する fallback も検討（複雑性アップなので避けたい） |
| 10 | **複数開発者が同時に Vast を起動 → DVC cache 競合** (ローカル `.dvc/cache` のシンボリックリンクがメインリポ共有のため、`dvc pull` lock が発生) | Low | Low | (a) Vast 側の cache はインスタンスローカル disk なので worktree 共有とは独立、競合しない。(b) ローカル側は `dev/vast-pull` 同時実行時のみ問題、シリアル実行を README で促す |
| 11 | **PR 上 main merge 漏れ** (採用したのに main に dvc.lock や eval メモ等を merge し忘れ) | Low | Low | (a) `dev/vast-promote` の最後に「次は git commit + PR 作成してください」のメッセージを表示。(b) PR template に「DVC 管理 weights を採用したか」のチェックリスト項目を追加（次回 PR template 改修時） |

## External Dependencies

- **Vast.ai**: API 可用性、価格変動、host availability。SLA は明示なし、SLO は practical で reliability ≥ 0.99 で絞れば実用可。
- **AWS S3 (ap-northeast-1)**: 既存。99.99% 可用性。
- **GitHub.com**: clone 元。public 不通は短期、private 化想定なら HTTPS PAT 認証。
- **PyPI / astral.sh (uv インストールスクリプト)**: Vast の onstart は外向き HTTPS が必要。Vast インスタンスは通常許可されている。

## Technical Debt

- **`weights_iter*.pt` の git 残置**: 旧 iter 群 (iter6〜iter15) は git 管理のまま。新運用で `weights_iter16.pt` 以降は作らず、すべて `runs/<run_id>/best.pt` に統一する方針。既存ファイルの DVC 移行は別 feature。
- **`train.py` の typer 依存**: 既に他 stage と同じパターンだが、CLI 引数として override パスを受け取る道もある。env 経由に倒すと、テスト時に env 設定を都度行う必要があり、単体実行性 (CLI からのデバッグ実行性) が下がる。これは妥協。
- **`vast/cli.py` のサブコマンド肥大**: 4 サブコマンド (train/pull/promote/cost-report) が同居。将来 `vast logs <id>`、`vast destroy <id>` 等が増えたら別ファイル分割を検討。
- **GPU 推論の Kaggle 提出への未対応**: `policy/agent.py` は CPU 推論前提。Kaggle Sandbox は GPU 不利用なので問題ないが、将来 Kaggle Notebook 提出になったら別 feature 必要。

## Open Items

- **Vast 起動時の "running instances safety check" の閾値**: `dev/vast-train` 実行時に既存 running instance が 0 でなければ警告するか、3 以上で警告するか、UX 確認要。 (Step 9 で実装、要相談)
- **`docs/experiment/vast_cost_report_*.md` の生成タイミング**: 手動実行 (`dev/vast-cost-report`) のみか、cron で自動か。本 feature では手動のみ、cron は将来 (`dev/schedule` skill 等) で対応。
- **Private GitHub repo の最終確認**: `git remote get-url origin` の出力で確認が必要。private なら GIT_PAT 経路を Step 7 の onstart テンプレに加える。
