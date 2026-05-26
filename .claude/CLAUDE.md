# Kaggle Orbit Wars: Bot Agents

Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) competition project. Develops AI agents on the `kaggle-environments` Orbit Wars environment supporting both 1v1 and 4-player FFA. See [`docs/competition/abstract.md`](../docs/competition/abstract.md) for the full competition spec.

## Agent Pipeline

1. observation (planets, fleets, comets, player, ...)
2. Feature Extraction (orbit prediction / threat assessment / production potential)
3. Policy (rule-based / learned model)
4. Action Selection [[from_planet_id, angle, num_ships], ...]
5. kaggle_environments env.step()

## Technology Stack

- **Language**: Python 3.13
- **Simulator**: `kaggle-environments` (Orbit Wars env)
- **Numerics**: NumPy, Pandas, Polars
- **AI / RL** (optional): PyTorch, Stable-Baselines3, etc. as needed
- **Testing**: Pytest + pytest-cov, Ruff, Mypy
- **Package Management**: UV

## Folder Structure

```
bot/                    Python implementation (pyproject.toml / uv.lock live here)
  src/                  Shared dev libs (submit, dataset, evaluate, utils, gpu/{vast,runpod,kaggle})
  pipeline/             Agent families only: rulebase / imitation / reinforce (reinforce/_bench は dev-only ベンチ)
  tests/                Pytest unit tests
simulator/              Orbit Wars simulator backends + adapter (bot 非依存の純粋シミュレータ層)
  python/               Apache-2.0 vendored copy of kaggle_environments/envs/orbit_wars (orbit_wars_vendor)
  rust/                 PyO3 + maturin Rust simulator (orbit_wars_rust._lib)
  jax/                  JAX-native reimplementation (orbit_wars_jax, jit + vmap, parity-tested)
  adapter/              Backend-agnostic env adapter (orbit_wars_sim, ORBIT_WARS_BACKEND で rust/python 切替)
infra/                  Terraform-based infrastructure (AWS, etc.)
  environment/          Per-environment root modules (dev / staging / prod)
  module/               Reusable shared modules
  runtime/              Container build assets (Dockerfile, buildspec, scripts)
data/                   4 layers (lake / processed / mart / output) (gitignored, symlinked to the main repo, DVC-managed)
  lake/                 Raw data (selfplay matches, kaggle_episodes, etc.)
  processed/            Pre-processed data
  mart/                 Curated data for training/evaluation (e.g. imitation/case1/train.parquet)
  output/               Generated artifacts (Vast.ai / RunPod GPU training models / Kaggle submission tar.gz + history / ablation aggregates)
dev/                    Development scripts (each cd's into bot and runs uv internally)
docs/
  competition/          Competition spec summaries (abstract.md, etc.)
  experiment/           Experiment plans and results — see `.claude/rules/docs.md` for naming
  plans/                Feature plans (one directory per feature, phases 00 → 06)
```

`uv run ...` is expected to run under `bot/`. From the repo root, use `dev/*` or `cd bot` first. See [`.claude/rules/command.md`](rules/command.md) for the command catalog (DVC, Vast.ai, RunPod, Kaggle submission policy).

## Glossary

| Term | Description |
|------|-------------|
| Orbit Wars | Kaggle-hosted simulation match competition. A modern take on Planet Wars |
| Planet | `[id, owner, x, y, radius, ships, production]`. Static or orbiting |
| Fleet | `[id, owner, x, y, angle, from_planet_id, ships]`. Speed depends on ship count |
| Comet | A group of 4 moving planets that appear at turns 50/150/250/350/450 |
| Home Planet | A player's initial planet (starts with 10 ships) |
| Skill Rating | Per-submission N(μ, σ²) rating. Updated only by win/loss |
| Overage Time | Extra thinking-time budget shared across each episode |

## Rules

| Rule file | Auto-loaded for | When to read manually |
|-----------|----------------|----------------------|
| `.claude/rules/python.md` | `**/*.py`, `**/*.ipynb` | Python language general rules |
| `.claude/rules/bot/pipeline.md` | `bot/pipeline/**` | Submit structure for case directories |
| `.claude/rules/bot/tests.md` | `bot/tests/**` | Pytest conventions |
| `.claude/rules/infra.md` | `infra/**` | Terraform / cloud infrastructure (AWS, etc.) |
| `.claude/rules/data.md` | `data/**` | data/ 4-layer structure (lake/processed/mart/output), DVC management, worktree symlink rules |
| `.claude/rules/command.md` | `dev/**` | Command catalog (`dev/*`, DVC, Vast.ai, Kaggle submission policy). Read on demand when running commands |
| `.claude/rules/docs.md` | `docs/**` | Documentation layout (docs/{competition,experiment,plans}) and naming rule |
| `.claude/rules/security.md` | Always loaded | Commits, secrets, CI/CD |

## Response Language And Interface

- Answer user questions concisely, organizing the response as a table, chart, list, short sentence, ASCII art, or similar structured format.
- Keep user-facing replies under 800 characters, excluding tables, charts, code blocks, and ASCII art (which can exceed the limit when needed).
- Use the `AskUserQuestion` tool when asking questions to the user
- Internal reasoning, tool calls, and intermediate notes: English.
- User-facing output (final replies, reports, summaries): Japanese.(全てのユーザー向けの出力は日本語で行うこと)
