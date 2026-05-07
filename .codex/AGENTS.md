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

```text
bot/                    Python implementation (pyproject.toml / uv.lock live here)
  src/
  pipeline/
  tests/                Pytest unit tests
simulator/              Orbit Wars simulator (official Python vendored copy + Rust reimplementation)
  python/               Apache-2.0 vendored copy of kaggle_environments/envs/orbit_wars
  rust/                 PyO3 + maturin Rust simulator (orbit_wars_rust._lib)
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
  experiment/           Experiment plans and results; see `.agents/rules/docs.md` for naming
  plans/                Feature plans (one directory per feature, phases 00 -> 06)
```

`uv run ...` is expected to run under `bot/`. From the repo root, use `dev/*` or `cd bot` first. See [`.agents/rules/command.md`](../.agents/rules/command.md) for the command catalog (DVC, Vast.ai, RunPod, Kaggle submission policy).

## Glossary

| Term | Description |
|------|-------------|
| Orbit Wars | Kaggle-hosted simulation match competition. A modern take on Planet Wars |
| Planet | `[id, owner, x, y, radius, ships, production]`. Static or orbiting |
| Fleet | `[id, owner, x, y, angle, from_planet_id, ships]`. Speed depends on ship count |
| Comet | A group of 4 moving planets that appear at turns 50/150/250/350/450 |
| Home Planet | A player's initial planet (starts with 10 ships) |
| Skill Rating | Per-submission N(mu, sigma^2) rating. Updated only by win/loss |
| Overage Time | Extra thinking-time budget shared across each episode |

## Rules

Codex does not auto-load `.agents/rules/*.md` by path. Before editing files in one of these scopes, read the matching rule file manually and apply it.

| Rule file | Applies to | When to read manually |
|-----------|------------|----------------------|
| `.agents/rules/python.md` | `**/*.py`, `**/*.ipynb` | Python language general rules |
| `.agents/rules/bot/pipeline.md` | `bot/pipeline/**` | Submit structure for case directories |
| `.agents/rules/bot/tests.md` | `bot/tests/**` | Pytest conventions |
| `.agents/rules/infra.md` | `infra/**` | Terraform / cloud infrastructure (AWS, etc.) |
| `.agents/rules/data.md` | `data/**` | data/ 4-layer structure, DVC management, worktree symlink rules |
| `.agents/rules/command.md` | `dev/**`, command execution | Command catalog, DVC, Vast.ai, RunPod, Kaggle submission policy |
| `.agents/rules/docs.md` | `docs/**` | Documentation layout and naming rule |
| `.agents/rules/security.md` | Always | Commits, secrets, CI/CD |

## Response Language And Interface

- Answer user questions concisely, organizing the response as a table, chart, list, short sentence, ASCII art, or similar structured format.
- Ask questions only when required to proceed safely; keep them concise.
- Internal reasoning, tool calls, and intermediate notes: English.
- User-facing output (final replies, reports, summaries): Japanese. All user-facing output must be in Japanese.
