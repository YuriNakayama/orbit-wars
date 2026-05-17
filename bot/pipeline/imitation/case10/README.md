# Imitation Case10 — filtered top80 1v1 BC

Case10 is derived from `imitation/case9` but keeps only the two active heads:

- `candidate_ships`
- `template_ships`

The case10 dataset filter is intentionally narrower for this experiment:

- match mode: `1v1`
- top submission filter: rating top 80 (`*_rating_mu`)
- turn count: 80–400 inclusive
- draw games: excluded

With the current Kaggle episode lake this selects 1,224 episodes.

## Commands

```bash
uv run --directory bot python -m pipeline.imitation.case10.training.preprocess \
  --config pipeline/imitation/case10/configs/il_case10_candidate.yaml

uv run --directory bot python -m pipeline.imitation.case10.training.train \
  --config pipeline/imitation/case10/configs/il_case10_candidate.yaml

uv run --directory bot python -m pipeline.imitation.case10.training.train \
  --config pipeline/imitation/case10/configs/il_case10_template.yaml
```

RunPod registry keys:

- `case10_candidate`
- `case10_template`
