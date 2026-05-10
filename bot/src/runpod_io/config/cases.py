"""Training case registry and path helpers for RunPod launches."""

from __future__ import annotations

from pathlib import Path

import typer

CASE_DEFAULTS: dict[str, dict[str, str]] = {
    "case1": {
        "stage": "train_imitation_case1",
        "train_module": "pipeline.imitation.case1.training.train",
        "config_arg": "",
        "preprocess_cmd": "",
        "canonical_weights": "bot/pipeline/imitation/case1/policy/weights.pt",
    },
    "case3": {
        "stage": "train_imitation_case3",
        "train_module": "pipeline.imitation.case3.training.train",
        "config_arg": "--config pipeline/imitation/case3/configs/il_phase2.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case3.training.preprocess "
            "--config pipeline/imitation/case3/configs/il_phase2.yaml"
        ),
        "canonical_weights": ("bot/pipeline/imitation/case3/policy/weights.pt"),
    },
    "case4": {
        "stage": "train_imitation_case4",
        "train_module": "pipeline.imitation.case4.training.train",
        "config_arg": "--config pipeline/imitation/case4/configs/il_case4.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case4.training.preprocess "
            "--config pipeline/imitation/case4/configs/il_case4.yaml"
        ),
        "canonical_weights": "bot/pipeline/imitation/case4/policy/weights.pt",
    },
    "case5": {
        "stage": "train_imitation_case5",
        "train_module": "pipeline.imitation.case5.training.train",
        "config_arg": "--config pipeline/imitation/case5/configs/il_case5.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case5.training.preprocess "
            "--config pipeline/imitation/case5/configs/il_case5.yaml"
        ),
        "canonical_weights": "bot/pipeline/imitation/case5/policy/weights.pt",
    },
    "case6": {
        "stage": "train_imitation_case6",
        "train_module": "pipeline.imitation.case6.training.train",
        "config_arg": "--config pipeline/imitation/case6/configs/il_case6.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case6.training.preprocess "
            "--config pipeline/imitation/case6/configs/il_case6.yaml"
        ),
        "canonical_weights": "bot/pipeline/imitation/case6/policy/weights.pt",
    },
    "case7": {
        "stage": "train_imitation_case7",
        "train_module": "pipeline.imitation.case7.training.train",
        "config_arg": "--config pipeline/imitation/case7/configs/il_case7.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case7.training.preprocess "
            "--config pipeline/imitation/case7/configs/il_case7.yaml"
        ),
        "canonical_weights": "bot/pipeline/imitation/case7/policy/weights.pt",
    },
    "case8": {
        "stage": "train_imitation_case8",
        "train_module": "pipeline.imitation.case8.training.train",
        "config_arg": "--config pipeline/imitation/case8/configs/il_case8.yaml",
        "preprocess_cmd": (
            "pipeline.imitation.case8.training.preprocess "
            "--config pipeline/imitation/case8/configs/il_case8.yaml"
        ),
        "canonical_weights": "bot/pipeline/imitation/case8/policy/weights.pt",
    },
    "case9_three_head": {
        "stage": "train_imitation_case9_three_head",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_three_head.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case9/policy/weights_three_head.pt"
        ),
    },
    "case9_candidate": {
        "stage": "train_imitation_case9_candidate",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_candidate.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case9/policy/weights_candidate.pt"
        ),
    },
    "case9_candidate_ships": {
        "stage": "train_imitation_case9_candidate_ships",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_candidate_ships.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case9/policy/weights_candidate_ships.pt"
        ),
    },
    "case9_template_ships": {
        "stage": "train_imitation_case9_template_ships",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_template_ships.yaml"
        ),
        "preprocess_cmd": "",
        "canonical_weights": (
            "bot/pipeline/imitation/case9/policy/weights_template_ships.pt"
        ),
    },
    "case9_dual": {
        "stage": "train_imitation_case9_dual",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": "--config pipeline/imitation/case9/configs/il_case9_dual.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "bot/pipeline/imitation/case9/policy/weights_dual.pt",
    },
    "case9_per_planet": {
        "stage": "train_imitation_case9_per_planet",
        "train_module": "pipeline.imitation.case9.training.train",
        "config_arg": (
            "--config pipeline/imitation/case9/configs/il_case9_per_planet.yaml"
        ),
        "preprocess_cmd": (
            "pipeline.imitation.case9.training.preprocess "
            "--config pipeline/imitation/case9/configs/il_case9_per_planet.yaml"
        ),
        "canonical_weights": (
            "bot/pipeline/imitation/case9/policy/weights_per_planet.pt"
        ),
    },
    "case10_candidate": {
        "stage": "train_imitation_case10_candidate",
        "train_module": "pipeline.imitation.case10.training.train",
        "config_arg": (
            "--config pipeline/imitation/case10/configs/il_case10_candidate.yaml"
        ),
        "preprocess_cmd": (
            "pipeline.imitation.case10.training.preprocess "
            "--config pipeline/imitation/case10/configs/il_case10_candidate.yaml"
        ),
        "canonical_weights": (
            "bot/pipeline/imitation/case10/policy/weights_candidate_ships.pt"
        ),
    },
    "case10_template": {
        "stage": "train_imitation_case10_template",
        "train_module": "pipeline.imitation.case10.training.train",
        "config_arg": (
            "--config pipeline/imitation/case10/configs/il_case10_template.yaml"
        ),
        "preprocess_cmd": (
            "pipeline.imitation.case10.training.preprocess "
            "--config pipeline/imitation/case10/configs/il_case10_template.yaml"
        ),
        "canonical_weights": (
            "bot/pipeline/imitation/case10/policy/weights_template_ships.pt"
        ),
    },
    # case0 = RunPod E2E smoke pipeline. NOT a real training case — the model
    # is a 200-param MLP on synthetic data, designed to finish in minutes so
    # the GPU basis itself can be verified end-to-end.
    "case0": {
        "stage": "train_imitation_case0",
        "train_module": "pipeline.imitation.case0.training.train",
        "config_arg": "--config pipeline/imitation/case0/configs/smoke.yaml",
        "preprocess_cmd": "",
        "canonical_weights": "",
    },
}


def case_subdir(case: str) -> str:
    """Map a registry key to the on-disk case subdirectory.

    case9 has head variants registered as separate keys (case9_three_head /
    case9_candidate / case9_candidate_ships / case9_template_ships /
    case9_dual) but they all share the same
    `data/output/models/imitation/case9/` tree on disk (same training
    pipeline, different head_mode). Strip the `_<variant>` suffix.
    """
    if case.startswith("case9_"):
        return "case9"
    if case.startswith("case10_"):
        return "case10"
    return case


def runs_root_for(case: str) -> Path:
    return Path(f"data/output/models/imitation/{case_subdir(case)}/runs")


def case_defaults(case: str) -> dict[str, str]:
    if case not in CASE_DEFAULTS:
        raise typer.BadParameter(
            f"unknown case={case!r}; supported: {sorted(CASE_DEFAULTS)}"
        )
    return CASE_DEFAULTS[case]


__all__ = ["CASE_DEFAULTS", "case_defaults", "case_subdir", "runs_root_for"]
