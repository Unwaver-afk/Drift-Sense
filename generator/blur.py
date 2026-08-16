"""
Imaging Blur / PSF Degradation
================================
Models the Modulation Transfer Function (MTF) of the SEM beam spot
as a Gaussian Point Spread Function (Postek & Vladár, 1998).

Supports:
    * Isotropic Gaussian blur (standard)
    * Anisotropic Gaussian blur (different σ_x vs σ_y)
    * Per-sample randomised blur strength

Search images receive heavier blur to reflect the coarser
10 nm/pixel acquisition mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.ndimage as ndimage


@dataclass
class BlurConfig:
    """Parameters controlling imaging blur."""

    sigma: float = 0.5               # Isotropic blur σ (pixels)
    sigma_y_ratio: float = 1.0       # σ_y / σ_x ratio (1.0 = isotropic)
    search_sigma_factor: float = 2.0  # Search blur = sigma * this


def random_blur_config(
    rng: np.random.Generator,
    difficulty: str = "medium",
) -> BlurConfig:
    """Sample randomised blur parameters for the given difficulty."""
    ranges = {
        "easy":    {"sigma": (0.2, 0.5),  "aniso": (0.9, 1.1), "search_f": (1.5, 2.0)},
        "medium":  {"sigma": (0.3, 0.8),  "aniso": (0.8, 1.2), "search_f": (1.8, 2.5)},
        "hard":    {"sigma": (0.5, 1.2),  "aniso": (0.7, 1.4), "search_f": (2.0, 3.0)},
        "extreme": {"sigma": (0.8, 2.0),  "aniso": (0.6, 1.6), "search_f": (2.5, 4.0)},
    }
    r = ranges.get(difficulty, ranges["medium"])

    return BlurConfig(
        sigma=rng.uniform(*r["sigma"]),
        sigma_y_ratio=rng.uniform(*r["aniso"]),
        search_sigma_factor=rng.uniform(*r["search_f"]),
    )


def apply_blur(
    image: np.ndarray,
    cfg: Optional[BlurConfig] = None,
    is_search: bool = False,
) -> np.ndarray:
    """
    Apply MTF-inspired Gaussian blur to a [0, 1] float32 image.

    Parameters
    ----------
    image : 2-D float32 array.
    cfg : BlurConfig — if None, defaults are used.
    is_search : If True, apply heavier blur (coarser acquisition).

    Returns
    -------
    Blurred image, float32.
    """
    if cfg is None:
        cfg = BlurConfig()

    sigma_base = cfg.sigma * (cfg.search_sigma_factor if is_search else 1.0)

    if sigma_base <= 0:
        return image.copy()

    sigma_x = sigma_base
    sigma_y = sigma_base * cfg.sigma_y_ratio

    return ndimage.gaussian_filter(
        image.astype(np.float32),
        sigma=(sigma_y, sigma_x),
    ).astype(np.float32)
