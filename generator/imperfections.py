"""
Controlled Structural Imperfections
=====================================
Adds physically-plausible manufacturing imperfections to rendered layouts:

* Line-width variation (lithographic dose non-uniformity)
* Pitch variation (overlay error)
* Missing / weak features (mask defects)
* Spatially-varying intensity modulation (beam current variation)

These are designed to produce **realistic imperfection, not random corruption**.
The imperfection budget scales with the difficulty level.

Note: Most imperfections are applied *inside* the DRAM/FinFET renderers
via their respective config dataclasses. This module provides the
high-level ``ImperfectionConfig`` and ``random_imperfection_config``
for the orchestrator to use when configuring renderers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImperfectionConfig:
    """
    Aggregate imperfection budget.

    These values are propagated to the layout renderer configs
    (``DRAMConfig.line_width_jitter``, etc.).
    """

    line_width_jitter: float = 0.0        # Fraction of nominal width (0–0.15)
    pitch_jitter: float = 0.0             # Fraction of nominal pitch (0–0.15)
    missing_feature_prob: float = 0.0     # Per-feature dropout probability
    intensity_variation: float = 0.0      # Max ± relative modulation


def random_imperfection_config(
    rng: np.random.Generator,
    difficulty: str = "medium",
) -> ImperfectionConfig:
    """Sample randomised imperfection budget for the given difficulty."""
    presets = {
        "easy": {
            "lwj": (0.0, 0.0),
            "pj":  (0.0, 0.0),
            "mfp": (0.0, 0.0),
            "iv":  (0.0, 0.0),
        },
        "medium": {
            "lwj": (0.01, 0.04),
            "pj":  (0.01, 0.04),
            "mfp": (0.0, 0.008),
            "iv":  (0.01, 0.04),
        },
        "hard": {
            "lwj": (0.03, 0.10),
            "pj":  (0.03, 0.08),
            "mfp": (0.005, 0.015),
            "iv":  (0.03, 0.08),
        },
        "extreme": {
            "lwj": (0.08, 0.15),
            "pj":  (0.06, 0.15),
            "mfp": (0.01, 0.025),
            "iv":  (0.06, 0.12),
        },
    }
    p = presets.get(difficulty, presets["medium"])

    def _sample(key: str) -> float:
        lo, hi = p[key]
        return rng.uniform(lo, hi) if hi > lo else lo

    return ImperfectionConfig(
        line_width_jitter=_sample("lwj"),
        pitch_jitter=_sample("pj"),
        missing_feature_prob=_sample("mfp"),
        intensity_variation=_sample("iv"),
    )


def apply_imperfections(
    dram_or_finfet_cfg,
    imperf: ImperfectionConfig,
):
    """
    Apply imperfection budget to a layout renderer config *in place*.

    Parameters
    ----------
    dram_or_finfet_cfg : DRAMConfig or FinFETConfig dataclass instance.
    imperf : ImperfectionConfig with the desired budget.

    Returns
    -------
    The same config object, mutated.
    """
    dram_or_finfet_cfg.line_width_jitter = imperf.line_width_jitter
    dram_or_finfet_cfg.pitch_jitter = imperf.pitch_jitter
    dram_or_finfet_cfg.missing_feature_prob = imperf.missing_feature_prob
    dram_or_finfet_cfg.intensity_variation = imperf.intensity_variation
    return dram_or_finfet_cfg
