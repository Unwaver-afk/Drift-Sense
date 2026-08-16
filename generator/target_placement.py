"""
Multi-Strategy Target Placement
=================================
Places the reference-image target within the search-image FOV
using configurable strategies:

1. **Uniform** — anywhere valid within the search image
2. **Edge** — deliberately near image boundaries
3. **Center** — within ±200 px of center (legacy behaviour)
4. **Hard periodic** — placed to maximise the number of near-identical candidates

Strategy selection is controllable per sample or randomised by difficulty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PlacementConfig:
    """Parameters controlling target placement."""

    strategy: str = "uniform"                # "uniform" | "edge" | "center" | "hard"
    margin_nm: float = 800.0                 # Minimum distance from edge in nm
    edge_bias_fraction: float = 0.15         # For "edge": place within this fraction of border
    center_sigma_nm: float = 500.0           # For "center": Gaussian σ around image center


# Strategy probabilities by difficulty
_STRATEGY_WEIGHTS = {
    "easy":    {"uniform": 0.3, "center": 0.7, "edge": 0.0, "hard": 0.0},
    "medium":  {"uniform": 0.4, "center": 0.3, "edge": 0.15, "hard": 0.15},
    "hard":    {"uniform": 0.25, "center": 0.15, "edge": 0.25, "hard": 0.35},
    "extreme": {"uniform": 0.15, "center": 0.05, "edge": 0.30, "hard": 0.50},
}


def random_placement_config(
    rng: np.random.Generator,
    difficulty: str = "medium",
) -> PlacementConfig:
    """Pick a placement strategy probabilistically based on difficulty."""
    weights = _STRATEGY_WEIGHTS.get(difficulty, _STRATEGY_WEIGHTS["medium"])
    strategies = list(weights.keys())
    probs = np.array([weights[s] for s in strategies])
    probs /= probs.sum()

    strategy = rng.choice(strategies, p=probs)

    margin = rng.uniform(600.0, 1000.0)  # Some margin variation

    return PlacementConfig(
        strategy=strategy,
        margin_nm=margin,
        edge_bias_fraction=rng.uniform(0.08, 0.20),
        center_sigma_nm=rng.uniform(300.0, 800.0),
    )


def compute_placement(
    rng: np.random.Generator,
    cfg: Optional[PlacementConfig] = None,
    search_fov_nm: float = 10000.0,
    ref_fov_nm: float = 1000.0,
) -> Tuple[float, float]:
    """
    Compute the center position (cx_nm, cy_nm) for the reference
    target within the search FOV.

    Parameters
    ----------
    rng : Random generator.
    cfg : PlacementConfig — if None, defaults (uniform) are used.
    search_fov_nm : Total search FOV in nanometres.
    ref_fov_nm : Reference FOV in nanometres.

    Returns
    -------
    (cx_nm, cy_nm) — center position in nanometres.
    """
    if cfg is None:
        cfg = PlacementConfig()

    # Absolute min/max to ensure the reference fits inside the search image
    half_ref = ref_fov_nm / 2.0
    lo = half_ref + cfg.margin_nm
    hi = search_fov_nm - half_ref - cfg.margin_nm
    lo = max(lo, half_ref + 50.0)  # Safety floor
    hi = min(hi, search_fov_nm - half_ref - 50.0)

    if lo >= hi:
        # Fallback: center
        return search_fov_nm / 2.0, search_fov_nm / 2.0

    strategy = cfg.strategy

    if strategy == "center":
        cx = search_fov_nm / 2.0 + rng.normal(0.0, cfg.center_sigma_nm)
        cy = search_fov_nm / 2.0 + rng.normal(0.0, cfg.center_sigma_nm)

    elif strategy == "edge":
        # Pick one edge (top/bottom/left/right) then place near it
        edge_band = cfg.edge_bias_fraction * search_fov_nm
        edge = rng.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            cx = rng.uniform(lo, hi)
            cy = rng.uniform(lo, min(lo + edge_band, hi))
        elif edge == "bottom":
            cx = rng.uniform(lo, hi)
            cy = rng.uniform(max(hi - edge_band, lo), hi)
        elif edge == "left":
            cx = rng.uniform(lo, min(lo + edge_band, hi))
            cy = rng.uniform(lo, hi)
        else:  # right
            cx = rng.uniform(max(hi - edge_band, lo), hi)
            cy = rng.uniform(lo, hi)

    elif strategy == "hard":
        # Place away from center to maximise periodic ambiguity —
        # the localization algorithm's center-prior will have many
        # closer candidates to choose from incorrectly
        cx = rng.uniform(lo, hi)
        cy = rng.uniform(lo, hi)
        # Bias away from center
        center = search_fov_nm / 2.0
        if abs(cx - center) < search_fov_nm * 0.15:
            cx = lo if rng.random() < 0.5 else hi
        if abs(cy - center) < search_fov_nm * 0.15:
            cy = lo if rng.random() < 0.5 else hi

    else:  # "uniform"
        cx = rng.uniform(lo, hi)
        cy = rng.uniform(lo, hi)

    # Clip to valid range
    cx = float(np.clip(cx, lo, hi))
    cy = float(np.clip(cy, lo, hi))

    return cx, cy
