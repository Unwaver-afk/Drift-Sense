"""
SEM Secondary-Electron Edge Brightening
========================================
Simulates the characteristic bright-edge contrast seen in SEM images,
caused by increased secondary-electron yield at topographical edges
(Goldstein et al., 2018, §12.3).

The edge enhancement strength is randomized per sample to prevent
the localization algorithm from overfitting to a single contrast profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.ndimage as ndimage


@dataclass
class EdgeConfig:
    """Parameters controlling SEM edge brightening."""

    edge_weight: float = 0.25       # Strength of edge enhancement (0 = none)
    edge_weight_min: float = 0.10   # Used only by random_edge_config
    edge_weight_max: float = 0.40   # Used only by random_edge_config


def random_edge_config(
    rng: np.random.Generator,
    difficulty: str = "medium",
) -> EdgeConfig:
    """Sample randomised edge-enhancement parameters."""
    ranges = {
        "easy":    (0.15, 0.30),
        "medium":  (0.10, 0.35),
        "hard":    (0.08, 0.40),
        "extreme": (0.05, 0.45),
    }
    lo, hi = ranges.get(difficulty, ranges["medium"])
    weight = rng.uniform(lo, hi)
    return EdgeConfig(edge_weight=weight, edge_weight_min=lo, edge_weight_max=hi)


def apply_edge_enhancement(
    image: np.ndarray,
    cfg: Optional[EdgeConfig] = None,
) -> np.ndarray:
    """
    Apply SEM-style edge brightening to a [0, 1] float32 image.

    Uses Sobel gradient magnitude to approximate the secondary-electron
    edge escape enhancement observed at topographical features.

    Parameters
    ----------
    image : 2-D float32 array in [0, 1].
    cfg : EdgeConfig — if None, defaults are used.

    Returns
    -------
    Edge-enhanced image, float32, clipped to [0, 1].
    """
    if cfg is None:
        cfg = EdgeConfig()

    img = image.copy().astype(np.float32)

    if cfg.edge_weight <= 0:
        return img

    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    grad_mag = np.hypot(gx, gy)

    g_max = grad_mag.max()
    if g_max > 0:
        grad_mag /= g_max

    img = img + cfg.edge_weight * grad_mag
    return np.clip(img, 0.0, 1.0).astype(np.float32)
