"""RunPod 上の GPU 学習を起動・管理するためのローカル CLI 基盤。

vast パッケージ (`backend/src/vast/`) と並走し、`backend/src/runpod_io/` を
独立 mirror 実装として配置する。RunPod 公式 SDK の module 名 `runpod` と
衝突するのを避けるため、内部パッケージ名は `runpod_io` とし、SDK は各
モジュールで `import runpod as runpod_sdk` の alias 経由で読む規約。

Public API:
    - auth: AWS credentials と RUNPOD_API_KEY のローカル loader
    - run_meta: vast.run_meta から re-export + RunPod 用 helper
    - offers: runpod.get_gpus + runpod.get_gpu の薄いラッパ
    - instance: onstart テンプレ render と create_pod
    - volumes: network volume CRUD (GraphQL 直叩き含む)
    - cost: run.json 集計と markdown レポート (RunPod run のみ filter)
    - cli: typer ベースのサブコマンド (train/pull/promote/cost-report/volume)
"""
