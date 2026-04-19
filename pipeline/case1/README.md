# Case1 Baseline — Orbit Wars 2026 Reinforce

Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) のベースラインエージェント。原典: [sigmaborov 氏の公開ノートブック (Public Score 928.5)](https://www.kaggle.com/code/sigmaborov/lb-897-orbit-wars-2026-reinforce) を Apache 2.0 に従い移植し、パッケージ化したもの。

## ディレクトリ構造

```
pipeline/case1/
├── baseline/
│   ├── agent.py          # build_world + agent(obs) エントリポイント
│   ├── main.py           # Kaggle submission 用の再エクスポート
│   ├── strategy.py       # plan_moves + 戦略ヘルパ
│   ├── core/             # config / types / geometry / physics / world_model
│   └── missions/         # snipe / reinforcement / crash_exploit
├── evaluation/
│   └── snapshot_update.py  # 観測/action スナップショット再生成
├── eda/
│   └── replay_viewer.py    # Jupyter 再生ビューア
├── configs/
│   └── baseline.yaml     # CONFIG を YAML 化 (参考値)
└── notebook/
    ├── lb-897-orbit-wars-2026-reinforce.ipynb
    └── kernel-metadata.json
```

## セットアップ

```bash
# 依存インストール
uv sync
# pygame/SDL ビルドで失敗する場合は事前に:
#   macOS : brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
#   Linux : sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev

# 上記でも失敗する環境向け (pygame は本 feature では未使用なので skip 可):
#   uv pip install --no-deps kaggle-environments
#   uv pip install numpy pandas polars pyarrow pydantic pyyaml rich typer \
#     python-dotenv pytest ruff mypy types-PyYAML jsonschema requests
#   uv pip install -e . --no-deps

# Kaggle CLI (ノートブック取得用、dev venv を汚さない)
uv tool install kaggle
```

### Kaggle 認証

以下のいずれかで設定:

- `~/.kaggle/kaggle.json` を配置し `chmod 600 ~/.kaggle/kaggle.json`
- 環境変数 `KAGGLE_USERNAME` / `KAGGLE_KEY` を設定

## ノートブック取得

```bash
# 最新 slug を確認
kaggle kernels list -s "orbit-wars-2026-reinforce"

# 取得 (slug は 2026-04 時点)
kaggle kernels pull sigmaborov/lb-897-orbit-wars-2026-reinforce \
  -p pipeline/case1/notebook/ -m
```

## 自己対局 (evaluation フレームワーク)

対戦実行は汎用フレームワーク `src/dataset/` に移管済み。以下のように呼ぶ:

```bash
# 1v1 × 5 エピソード (seed 固定)
uv run python -m dataset run \
  --agents baseline_v1,baseline_v1 --mode 1v1 -n 5 --seed 0 --parallel 4

# 4P FFA × 10 エピソード
uv run python -m dataset run \
  --agents baseline_v1,baseline_v1,baseline_v1,baseline_v1 \
  --mode ffa4 -n 10 --parallel 4

# 集計 / 一覧
uv run python -m dataset list --mode 1v1 --limit 10
```

結果は `data/lake/selfplay/matches/index.parquet/mode={1v1,ffa4}/...` (Parquet hive) と
`data/lake/selfplay/matches/replays/{match_id}.json.gz` に保存される (`data/` は gitignore)。

## テスト

```bash
uv run ruff format --check pipeline/case1 tests/pipeline/case1
uv run ruff check pipeline/case1 tests/pipeline/case1
uv run mypy pipeline/case1
uv run pytest tests/pipeline/case1 -v
# 高速レーン (snapshot を除外)
uv run pytest tests/pipeline/case1 -v -m "not slow"
```

カバレッジは本 feature では `--cov=src` のため pipeline/case1 は測定対象外。代わりに snapshot 一致性で品質を担保する。

## 品質ゲート

- **ノートブック挙動一致** — `tests/pipeline/case1/snapshots/obs_seed0_turn10.json` + `action_seed0_turn10.json` に固定したターン観測と action の diff が 0 件。環境本体は seed 固定でも完全再現ではないため、`obs` を固定してエージェントの決定性を担保している。再生成は `uv run python -m pipeline.case1.evaluation.snapshot_update`。
- **DONE 到達** — `env.run([agent, agent])` が例外なく完走する。
- **タイムアウト** — `python -m dataset run` の Summary に表示される `turn_p95 < 1.0s`。

## ライセンス

原典ノートブックは Apache License 2.0 のもと公開されている。本ディレクトリの `baseline/LICENSE` にライセンス本文を配置し、全 Python ファイル先頭に出典と帰属コメントを付与する。

## 既知の制約

- `uv sync` は pygame/SDL ビルドでコケることがある。macOS/Linux ともにネイティブ SDL2 を先にインストールすること。
- Kaggle ノートブックの slug はスコア更新時に変わる可能性がある (例: `orbit-wars-2026-reinforce` → `lb-897-orbit-wars-2026-reinforce`)。上記の `kaggle kernels list` で追跡する。
- `src/` 側モジュール (`src/agents/`, `src/features/` 等) への分割は本 feature のスコープ外 (後続 Step B)。
