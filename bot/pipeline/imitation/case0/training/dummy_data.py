"""Deterministic synthetic dataset for the case0 RunPod smoke pipeline.

Generates `num_samples` rows of `(features[in_dim] float32, label int64)` and
writes them as a parquet file. Output is deterministic given `seed`, verified
via SHA-256 in tests so CI catches accidental nondeterminism.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import typer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DummyDatasetSpec:
    num_samples: int = 32
    in_dim: int = 8
    num_classes: int = 4
    seed: int = 0


def generate(spec: DummyDatasetSpec) -> pd.DataFrame:
    """Build the dataset in memory. Uses `numpy.random.default_rng(seed)`."""
    rng = np.random.default_rng(spec.seed)
    features = rng.standard_normal(
        size=(spec.num_samples, spec.in_dim), dtype=np.float32
    )
    labels = rng.integers(
        low=0, high=spec.num_classes, size=spec.num_samples, dtype=np.int64
    )
    df = pd.DataFrame(
        {f"f{i}": features[:, i] for i in range(spec.in_dim)} | {"label": labels},
    )
    return df


def write(spec: DummyDatasetSpec, out_path: Path) -> Path:
    """Write parquet and return the path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate(spec)
    df.to_parquet(out_path, index=False)
    logger.info("wrote %d rows to %s", len(df), out_path)
    return out_path


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


app = typer.Typer(add_completion=False, help="Generate case0 dummy dataset.")

_OUT_OPTION = typer.Option(..., "--out", help="Output parquet path.")
_NUM_SAMPLES_OPTION = typer.Option(32, "--num-samples")
_IN_DIM_OPTION = typer.Option(8, "--in-dim")
_NUM_CLASSES_OPTION = typer.Option(4, "--num-classes")
_SEED_OPTION = typer.Option(0, "--seed")


@app.command()
def main(
    out: Path = _OUT_OPTION,
    num_samples: int = _NUM_SAMPLES_OPTION,
    in_dim: int = _IN_DIM_OPTION,
    num_classes: int = _NUM_CLASSES_OPTION,
    seed: int = _SEED_OPTION,
) -> None:
    spec = DummyDatasetSpec(
        num_samples=num_samples,
        in_dim=in_dim,
        num_classes=num_classes,
        seed=seed,
    )
    path = write(spec, out)
    typer.echo(f"OK: {path} sha256={sha256_of_file(path)[:12]}")


if __name__ == "__main__":
    app()
