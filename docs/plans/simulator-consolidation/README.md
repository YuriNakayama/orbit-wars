# Simulator Consolidation & bot/ Structure Refactor

**実施日**: 2026-05-26

`bot/` 内に散らばっていたシミュレータ系コードを `simulator/` 配下に集約し、
`simulator/` を「bot に依存しない純粋なシミュレータ層」として完結させる。
あわせて `bot/scripts/` を廃止して dev-only ベンチを `pipeline/_bench/` へ寄せた。

## 背景

- JAX env (`bot/src/jax_env/`) と env adapter (`bot/src/env/`) が `bot/` 内に
  あり、公式 Python sim (`simulator/python`) / Rust sim (`simulator/rust`) と
  非対称だった。
- adapter は `utils.repo_root` に依存し、`simulator/` 単体では完結しなかった。
- `bot/scripts/` は dev 専用ベンチ 3 本のみで、`pipeline/_bench/` と役割が重複。

## 変更後の構成

```
simulator/                  ← bot 非依存の純粋シミュレータ層
  python/   orbit_wars_vendor   公式 Python sim (Apache-2.0 vendored)
  rust/     orbit_wars_rust     PyO3 + maturin Rust sim
  jax/      orbit_wars_jax      ★JAX-native sim (旧 bot/src/jax_env)
  adapter/  orbit_wars_sim      ★backend 切替 adapter (旧 bot/src/env)

bot/
  src/        submit / dataset / evaluation / utils / gpu/{vast,runpod,kaggle}
  pipeline/   rulebase / imitation / reinforce ← 3 family のみ
    reinforce/_bench/   ★dev-only ベンチ (旧 bot/scripts + 旧 pipeline/_bench を吸収)
  tests/
```

★ = 本リファクタで移動。各々 editable な uv パッケージ
(`orbit-wars-jax` / `orbit-wars-sim`) として `bot/pyproject.toml` の
`[tool.uv.sources]` から参照。

`reinforce/_bench` の 3 つの `*_gpu/run_bench.py` 間で重複していた GPU/JAX
RunPod 用インフラヘルパ (cuda12 install / jax reload / run dir / device
診断) は `bot/src/utils/gpu_bench.py` に統合し、drift を解消 (reload は
`extra_prefixes` で parameterize)。BenchResult / `_bench_*` はベンチ固有の
ため各モジュールに残置。

## GPU CLI の `gpu/` 集約 (採用)

GPU プロバイダ CLI 3 本を `bot/src/gpu/` 配下へ集約 (ユーザー決定):

| 旧 | 新 | CLI |
|----|----|-----|
| `src/runpod_io` | `src/gpu/runpod` | `python -m gpu.runpod` |
| `src/vast` | `src/gpu/vast` | `python -m gpu.vast` |
| `src/kaggle_kernel` | `src/gpu/kaggle` | `python -m gpu.kaggle` |

`-m` CLI が `gpu.*` に変わるため、`dev/{runpod,vast,kaggle}` wrapper (kaggle-kernel→
kaggle にリネーム)、RunPod onstart の `60_train.sh` (`-m gpu.runpod.execution.
system_monitor`)、Kaggle kernel template の埋め込み import (`gpu.kaggle.interactive.
channel`) など **リモート実行される文字列も全て追従**。`vast` の rename は 3rd-party
`vastai` / `vast.ai` 文字列を巻き込まないよう厳密な正規表現で実施。

## 採用しなかった案

- **`bot/src/` 残り 4 パッケージ (submit/dataset/evaluation/utils) を `orbit_wars_bot`
  単一 namespace へ統合**: `python -m submit` 等の CLI エントリ規約と衝突し、
  silent breakage リスクが高いため**見送り** (ユーザー決定)。GPU 系のように明確な
  グルーピング理由がある場合のみ集約する方針。

## キーとなる設計判断

| 判断 | 理由 |
|------|------|
| adapter を `simulator/adapter/` に独立させ rust/python の**中には置かない** | adapter は複数 backend を束ねる上位層。特定 backend 内に置くと逆転依存になる |
| adapter の `find_repo_root` 依存を `__file__` 相対 (`parents[2]`) へ置換 | `simulator/` を bot 非依存にし、worktree でも repo-root walk 不要に |
| JAX/adapter の test は `bot/tests/` に据え置き | parity test が vendor/adapter と密結合。`simulator/{jax,adapter}` に test dir は作らず、`dev/test-bot` は import smoke のみ追加 |
| `_bench/` 移動で `Path(__file__).parents[N]` を +1 補正 | 1 階層深くなった分の path 解決ずれを修正 |

## 検証

- `dev/test-bot` 相当: ruff format/check + mypy (1055 files) + pytest 全グリーン
- baseline (Phase 0): unit 1735 passed / 3 skipped
- Phase 1 後: jax_env + parity 705 passed
- Phase 2 後: rust/python 両 backend の env 生成 + episode 実行を機能確認、
  adapter-touching 126 passed
- Phase 3 後: 3 bench module import + featurizer bench 実行 smoke OK

## コミット

1. `:truck: JAX env を simulator/jax/orbit_wars_jax へ集約`
2. `:truck: env adapter を simulator/adapter/orbit_wars_sim へ (bot非依存化)`
3. `:truck: bot/scripts を廃止し pipeline/_bench へ集約`
