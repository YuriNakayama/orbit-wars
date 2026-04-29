# artifacts/

Vast.ai 等の GPU 学習で生成された **候補モデル** の保存先。`policy/weights.pt`（Kaggle 提出の正本）とは分離して管理する。

## ディレクトリ構造

```
artifacts/
└── models/
    └── imitation/
        └── case1/
            └── runs/
                └── <run_id>/
                    ├── best.pt        # 学習済み weights (DVC 管理)
                    ├── metrics.json   # epoch 履歴 + best metrics
                    └── run.json       # lineage record (git SHA / params hash / GPU 等)
```

`<run_id>` の形式: `<YYYYMMDD-HHMMSS>__<branch_slug>__<sha7>__seed<N>`

## 運用

- ディレクトリ全体は `.gitignore` で **git 管理外**。`*.dvc`、`*.md`、`.gitkeep` のみ git で追跡。
- 各 `runs/<run_id>/` は **`dvc add` で個別管理** され、S3 remote (`s3://orbit-wars-dvc-286854171013/remote`) に push される。
- ローカルで取得するときは `dev/vast pull <run_id>` を使う。
- 採用するときは `dev/vast promote <run_id>` を実行し、`best.pt` を `backend/pipeline/imitation/case1/policy/weights.pt` にコピーする。

詳細は [`docs/plans/vast-ai-basis/`](../docs/plans/vast-ai-basis/) を参照。
