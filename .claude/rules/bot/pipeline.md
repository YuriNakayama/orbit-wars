---
paths:
  - "bot/pipeline/**"
---

# Pipeline Experiment Rules

Conventions for working in `bot/pipeline/<category>/case*/` directories — directory layout, experiment discipline, long-running training, and evaluation. `<category>` currently has three families: `rulebase/`, `imitation/`, and `reinforce/`. Case numbers are assigned **independently per category** starting from 1 (rulebase/case1 and imitation/case1 are unrelated).

> **Submit constraints live in [`submit.md`](submit.md).** The Kaggle-runnable structure (`main.py` entrypoint, relative imports, `.submitignore`, dry-run verification) is loaded only when you edit `bot/src/submit/**` or run `dev/submit`. When creating or editing a submittable case, read `submit.md`.

Path notation below uses `bot/` as the anchor (`pipeline/<category>/case<N>/...`). `uv run ...` / `dev/submit ...` are expected to execute with `bot/` as the working directory.

## Directory layout

### Per-case layout

```
case<N>/
├── main.py            # Kaggle entrypoint (thin wrapper — see submit.md)
├── __init__.py
├── README.md          # purpose, strategy summary, latest publicScore (if known)
└── baseline/  or  policy/   # agent body — must export `agent` callable
```

- `pipeline/<category>/case<N>/main.py` is always an entrypoint. Keep it as a thin wrapper of roughly 20 lines. Do not put business logic in it. (For the entrypoint template and import rules, see [`submit.md`](submit.md).)
- The implementation lives in subpackages under `pipeline/<category>/case<N>/<package>/` (e.g. `baseline/`, `policy/`). Maintain the hierarchy for readability and maintainability.
- Auxiliary directories such as `evaluation/`, `configs/`, `eda/`, `notebook/` may sit under `pipeline/<category>/case<N>/`. They are harmless on Kaggle as long as they are not imported from `main.py`, but the tar.gz size should still be kept small.

### Cross-case independence rule

Cases must remain self-contained. **Never** import from another case (`from pipeline.rulebase.case2.* import ...` inside `case1/`). When the same helper is needed in multiple cases, copy it into each — duplication is preferred to cross-case coupling. Shared development utilities live in `bot/src/` (e.g. `src/evaluate/`, `src/utils/repo_root.py`) and are imported only from `evaluation/` / `training/` (excluded from submission tar).

## Experiment discipline

**Change only what you are testing; hold everything else fixed.** An experiment exists to test one thing (a parameter, a method, a dataset, ...). Fix every other variable and vary only the subject under test, then compare. Always state up front what is being tested, and never change multiple items at once in a way that makes the result impossible to attribute — if two things changed, you cannot tell which one moved the metric.

## Long-running training operations

For anything that runs long (model training: imitation / reinforce on RunPod / Kaggle / Vast), assume the run **can fail or be preempted at any moment** and design so that partial progress always survives.

### Crash-safe checkpointing (mandatory)

- Upload the model **weights, logs, and training-time accuracy / metrics to S3 on a suitable cadence** (e.g. every epoch / every iter), not only on a clean finish. A failed run must still leave (a) the latest weights as a backup, (b) enough log to diagnose the failure cause, and (c) the partial training curve to inspect progress.
- Append metrics incrementally (e.g. `metrics.json` per iter) and update `best.pt` whenever a new best appears, so the in-flight state is observable at any time. Pass an env var such as `ORBIT_WARS_BEST_S3_PREFIX` to the train script and call `s3.upload_file()` on each new best.

### Dry-run before the real run (mandatory)

Before launching the full-length run, do a **simplified run** (reduced epochs / iters) and confirm end to end that:

- the behavior, logs, and artifact upload work as expected,
- in-flight health (progress / steps / loss / GPU·CPU·memory) is observable **at any arbitrary moment during the run**,
- training-time statistics (accuracy / loss / reward etc.) can be read mid-run,
- and **on failure, all of the above are still persisted to S3**.

Only after this simplified run passes should the full-length run be launched.

### GPU acquisition (RunPod etc.)

When the target GPU is unavailable, **poll-and-acquire in a tight loop**: check availability of the intended GPU and try to acquire, repeatedly. RunPod frees capacity momentarily, so a fast check-and-grab loop usually succeeds eventually. Run this loop fast. If the GPU cannot be acquired within ~30 minutes, switch the target to a different GPU (the original may become available again later, so it can be retried afterward). A pod that is never created incurs zero cost, so backoff-and-retry is safe.

### Reinforcement-learning progress check (mandatory for RL)

When building a model with reinforcement learning, run a match against a **fixed opponent** (e.g. a rulebase agent) once every N epochs, so you can verify whether learning is actually progressing. Self-play reward trending up is **not** sufficient evidence — a fixed-opponent win-rate that moves over time is what confirms real progress (see the train/eval-parity pitfalls where self-play improvement did not transfer to a real opponent).

## Anti-patterns

- Cross-case imports (`from pipeline.rulebase.case2.baseline import ...` inside `case1/`) — violates case independence.
- Changing more than the variable under test in one experiment — the result becomes impossible to attribute.
- Hardcoding paths as typer-Option defaults (`Path("data/.../foo.parquet")`) without making them overridable — use `--config`/`--out` / params.yaml hooks so the script is testable in isolation.

## Evaluation metric interpretation

Kaggle Orbit Wars publicScore is a **relative metric** computed against other participants' submissions, and the opponent pool drifts over time, so **the same agent can produce very different publicScores depending on submission timing**. Do not use Kaggle-side numbers to judge the merit of a change. **Evaluate agents exclusively on local match results.**
