# runpod-basis — Risks and Dependencies

## Risk List

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|------------|
| 1 | **RunPod pod が destroy されず課金高騰** (self-destroy trap が機能せず、人間が気付かないまま GPU を放置) | High | Medium | (a) onstart 末尾の `trap EXIT` で **成功時のみ** `runpodctl stop pod`。(b) 冒頭で **2h タイムアウト保険**: `( sleep 7200 && runpodctl stop pod "$INSTANCE_ID" ) &` を background 投下、trap 失敗時の最終防衛。(c) `dev/runpod cost-report` を定期実行可能にし起動中 pod もダッシュボード化。(d) `dev/runpod train` 起動時に既存 pod 一覧を表示する safety check を Step 10 で追加。(e) RunPod の Account Settings で max spend 通知を設定するよう README に明記 |
| 2 | **AWS / RUNPOD_API_KEY の pod 側漏洩** (ssh 他人アクセス、ログ流出、デバッグ時の env ダンプ) | High | Low | (a) `set -x` を絶対に有効化しない (onstart は `set -euo pipefail` のみ)。(b) `env >> /etc/environment` は pod 内 ssh 用、stdout には流さない。(c) IAM ポリシーは `s3:DeleteObject` 含まず破壊不可 (vast 同方針)。(d) RunPod の API key は **pod-scoped** な `runpodctl` キーと **アカウント全権** な `RUNPOD_API_KEY` の 2 種ある。**onstart に渡すのは前者のみで充分**だが、本基盤では pod 起動コードに `RUNPOD_API_KEY` を渡している (`runpodctl` の認証経路解決のため)。漏洩したら個人アカウント全 pod に影響なので **API key を pod 内で `unset RUNPOD_API_KEY` してから onstart 本体を実行する** か、**onstart の env リストから `RUNPOD_API_KEY` を外して pod-scoped runpodctl で十分か検証** を Step 6 で確認。 |
| 3 | **`dvc push` 失敗 → 成果物消失** (S3 一時障害、network 断、disk full、credentials 失効) | High | Low | (a) onstart の trap で **失敗時は self-destroy しない**。(b) `dvc push` の戻り値検査 → 失敗で `status=failed` を `run.json` に書き、destroy せず exit 1。(c) `runpodctl pod logs` で原因確認 → `runpodctl pod ssh <id>` (or 標準 ssh) で再 push 可能。(d) `--container-disk 40` で disk 余裕。 |
| 4 | **train.py が canonical `policy/weights.pt` を誤上書き** (env 脱落、テスト不足) | High | Low | (a) train.py の防御弾: `ORBIT_WARS_RUNPOD_POD_ID` と `ORBIT_WARS_VAST_INSTANCE_ID` の **両方 set はエラー**、片方が set されていれば対応する provider field を埋める。`ORBIT_WARS_RUN_DIR` が無いのに `ORBIT_WARS_RUNPOD_POD_ID` が set のときも assertion で fail。(b) Step 2 の test で env 組み合わせを網羅的にカバー。(c) 万一上書きされても `dvc commit` 前なら `dvc checkout policy/weights.pt` で復旧可能。 |
| 5 | **RunPod pod の host 不安定 (Community Cloud)** | Medium | Medium | (a) デフォルト `--cloud-type=SECURE` で T3/T4 DC を使う。(b) Community 選択時は README で明示。(c) 1 run < 30 分なので影響範囲小。 |
| 6 | **`runpod` SDK のバージョン互換破壊** (RunPod 側 API 変更で SDK が動かない) | Medium | Low | (a) `pyproject.toml` で `runpod>=1.7.0,<2.0.0` と pin。(b) e2e 成功 commit を mark しておき、SDK 更新時は手動 release tag で記録 |
| 7 | **GPU 学習結果が CPU/Vast 学習結果と差異** (mixed precision、CUDA 非決定論) | Medium | Medium | (a) `torch.use_deterministic_algorithms(False)` のまま許容 (vast 同方針)。(b) `run.json.train_metrics.device` を残し評価フェーズで「CPU / 異 provider GPU は同条件で比較不可」と明示。(c) canonical を再現する場合は **CPU で再学習** する逃げ道を維持。 |
| 8 | **Network volume の DC 制約による pod 立ち上げ失敗** (volume が乗っている DC で pod が capacity 不足) | Medium | Medium | (a) volume 作成時に DC を memo (run.json.runpod_offer_snapshot.data_center_id)。(b) pod 起動失敗時 actionable error: `"no SECURE pods available with <gpu> in <dc> — try --cloud-type=COMMUNITY (volume 不可) or change network volume region"`。(c) Phase 2 で `dev/runpod volume create` 時に複数 DC で並行作成する戦略を検討。 |
| 9 | **`git clone` 認証エラー** (private repo + PAT 必要) | Medium | Medium | (a) onstart は `https://github.com/<user>/orbit-wars.git` の HTTPS clone を試す。(b) private なら `--env GIT_PAT=...` で PAT を注入し URL に埋め込む (vast 同設計)。(c) public/private の最終確認は実装着手時。 |
| 10 | **`uv sync --locked` がリモート PyTorch CUDA wheel 取得で失敗** | Medium | Medium | (a) base image を `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` 固定、torch は image に既存。(b) `uv sync` 3 回 retry。(c) image 焼込みで cache hit する場合は wheel download なしでほぼゼロ秒。 |
| 11 | **パッケージ命名 `runpod_io` と SDK `runpod` の混乱** | Low | Low | (a) `__init__.py` の docstring で `import runpod as runpod_sdk` 規約を明記。(b) コードレビュー時にチェックリストとして「自パッケージ import が `from runpod_io.X import Y` または `from .X import Y`、SDK が `import runpod as runpod_sdk` であることを確認」を追加。(c) `dev/test-backend` の `python -c "import runpod_io; import runpod"` smoke を Step 3 acceptance に含める。 |
| 12 | **`runpodctl` が pod 内で見つからない** (image によっては未 install) | Medium | Low | (a) 推奨 image (`runpod/pytorch:...`) には pre-install 済み。(b) onstart 内で `command -v runpodctl` チェックし、見つからなければ self-destroy をスキップして `echo` で警告 → 2h タイムアウト保険が最終防衛。(c) image 変更時は手動検証必須を README に明記。 |
| 13 | **`dev/runpod` と `dev/vast` の混同** (誤って provider を切替えてしまう) | Low | Low | (a) 各 CLI の起動メッセージに `[runpod]` / `[vast]` の prefix を入れる。(b) `run.json` の field でどちら経由か追跡可能。(c) cost-report 出力 path で provider 別 (`runpod_cost_report_*.md` / `vast_cost_report_*.md`) なので混在しない。 |
| 14 | **複数開発者が同時に RunPod pod 起動 → DVC cache 競合** | Low | Low | (a) Pod 内 cache は instance ローカル disk (network volume 切替時を除く)。(b) ローカル `.dvc/cache` は worktree 共有なので `dev/runpod pull` 同時実行は控える運用ルールあり (`.claude/rules/command.md`)。 |
| 15 | **PR 上 main merge 漏れ** (採用したのに dvc.lock や eval メモを merge し忘れ) | Low | Low | (a) `dev/runpod promote` の最後に「次は git commit + PR 作成してください」のメッセージ表示 (vast 同等)。(b) PR template に「DVC 管理 weights を採用したか」のチェックリスト項目を追加。 |

## External Dependencies

- **RunPod**: API 可用性、価格変動、host availability。SLA は明示なし、SLO は practical で Secure Cloud であれば実用可。
- **AWS S3 (ap-northeast-1)**: 既存 (vast と共有)。99.99% 可用性。
- **GitHub.com**: clone 元 (両基盤で同じ private repo)。HTTPS PAT 認証。
- **PyPI / astral.sh**: pod 起動時に外向き HTTPS 必要。RunPod pod 通常許可。
- **`runpod` Python SDK**: 公式、週次 release。1.7+ 必須。
- **`runpodctl` CLI**: 公式 RunPod image に pre-install。

## Technical Debt

- **`vast.run_meta` への RunPod field 追加**: vast パッケージに provider 中立な拡張を入れているので、Phase 2 で `cloud/run_meta.py` のような中立 module への切り出しを検討。
- **Volume API の GraphQL 直叩き**: SDK が薄いため `run_graphql_query` を使用。SDK が将来 wrapper を出したら差し替え。
- **`runpod_io.auth` から `vast.auth` を import**: 共通化なら別 module 推奨。Phase 2 で `cloud/auth.py` 切り出し候補。
- **Spot/Interruptible 非対応**: 1 run < 30 分なので不要だが、長時間 RL 学習時は spot bid を実装する別 feature が必要。
- **GPU data center filter**: `data_center_id` を pod 起動時に固定する必要があるが、現時点で UX として「volume の DC を自動取得」までは実装しない (手動指定)。Phase 2 で auto-resolve を追加。

## Open Items

- **Public/Private repo の最終確認**: `git remote get-url origin` で確認、private なら `GIT_PAT` 経路を Step 6 の onstart テンプレに加える (vast 同等処理がコピーされる前提)。
- **`RUNPOD_API_KEY` を pod に渡すべきか否か**: 現状 (Step 6 案) では渡しているが、`runpodctl` は pod-scoped key で動作するため、本当に必要か Step 6 の手動検証で決める。不要なら `unset RUNPOD_API_KEY` を onstart 末尾に追加し漏洩リスクを下げる。
- **Network volume の data center 自動取得**: pod 起動時に `volume.data_center_id` を opener に流し込む UX (`--data-center-id` を省略可能にする) は Step 10 で auto-resolve を実装するかどうか判断。
- **`docs/experiment/runpod_cost_report_*.md` の生成タイミング**: 手動実行のみ、cron は将来 (`/schedule` skill 経由)。
- **`dev/runpod` 起動時の running pod safety check の閾値**: 0/3 等の閾値、UX は Step 10 で要相談 (vast の同 open item と同期)。
