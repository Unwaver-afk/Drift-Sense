"""
Multi-Model SEM Sensor Noise
=============================
Implements physically-motivated, independent noise for SEM image simulation:

1. **Poisson shot noise** — electron beam counting statistics
   (Reimer, 1998, Ch. 4-5)
2. **Gaussian readout noise** — transimpedance amplifier thermal/Johnson noise
   (Reimer, 1998)
3. **Dead-pixel / salt-and-pepper noise** — detector defects
   (optional, for extreme difficulty)

All noise is applied *independently* per call — never shared between
reference and search images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NoiseConfig:
    """Parameters controlling noise application."""

    # Poisson shot noise
    shot_noise_scale: float = 0.03    # Higher → more noise (1/sqrt(electrons))
    # Gaussian readout noise
    thermal_sigma: float = 0.02       # Std-dev of additive Gaussian
    # Dead-pixel noise
    dead_pixel_prob: float = 0.0      # Probability per pixel of being dead
    dead_pixel_value: float = 0.0     # Intensity of dead pixels (0=black, 1=white)
    # Salt-and-pepper
    salt_pepper_prob: float = 0.0     # Combined probability of salt or pepper
    # Search-image amplification factors
    search_shot_factor: float = 2.2   # Shot noise is this × worse for search
    search_thermal_factor: float = 1.8  # Readout noise is this × worse for search


def random_noise_config(
    rng: np.random.Generator,
    difficulty: str = "medium",
) -> NoiseConfig:
    """Sample randomised noise parameters for the given difficulty."""

    ranges = {
        "easy":    {"shot": (0.015, 0.030), "therm": (0.010, 0.020), "dp": 0.0, "sp": 0.0},
        "medium":  {"shot": (0.025, 0.045), "therm": (0.015, 0.030), "dp": 0.0, "sp": 0.0},
        "hard":    {"shot": (0.040, 0.070), "therm": (0.025, 0.050), "dp": 0.0005, "sp": 0.001},
        "extreme": {"shot": (0.060, 0.100), "therm": (0.040, 0.070), "dp": 0.001, "sp": 0.002},
    }
    r = ranges.get(difficulty, ranges["medium"])

    return NoiseConfig(
        shot_noise_scale=rng.uniform(*r["shot"]),
        thermal_sigma=rng.uniform(*r["therm"]),
        dead_pixel_prob=r["dp"],
        salt_pepper_prob=r["sp"],
        search_shot_factor=rng.uniform(1.8, 2.6),
        search_thermal_factor=rng.uniform(1.5, 2.2),
    )


def apply_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    cfg: Optional[NoiseConfig] = None,
    is_search: bool = False,
) -> np.ndarray:
    """
    Apply independent SEM sensor noise to a [0, 1] float32 image.

    Parameters
    ----------
    image : 2-D float32 array in [0, 1].
    rng : Random generator (guarantees independence between calls).
    cfg : NoiseConfig — if None, defaults are used.
    is_search : If True, apply amplified noise (lower dwell time).

    Returns
    -------
    Noised image, float32, clipped to [0, 1].
    """
    if cfg is None:
        cfg = NoiseConfig()

    img = image.copy().astype(np.float64)

    # Scale noise for search images
    shot_scale = cfg.shot_noise_scale * (cfg.search_shot_factor if is_search else 1.0)
    therm_sig = cfg.thermal_sigma * (cfg.search_thermal_factor if is_search else 1.0)

    # 1. Poisson shot noise
    if shot_scale > 0:
        nominal_electrons = 1.0 / (shot_scale ** 2 + 1e-9)
        poisson_input = np.maximum(img, 0.01) * nominal_electrons
        img = rng.poisson(poisson_input).astype(np.float64) / nominal_electrons

    # 2. Gaussian readout noise
    if therm_sig > 0:
        img += rng.normal(0.0, therm_sig, img.shape)

    # 3. Dead-pixel noise
    if cfg.dead_pixel_prob > 0:
        dead_mask = rng.random(img.shape) < cfg.dead_pixel_prob
        img[dead_mask] = cfg.dead_pixel_value

    # 4. Salt-and-pepper noise
    if cfg.salt_pepper_prob > 0:
        sp_rand = rng.random(img.shape)
        half_p = cfg.salt_pepper_prob / 2.0
        img[sp_rand < half_p] = 0.0          # pepper
        img[sp_rand > (1.0 - half_p)] = 1.0  # salt

    return np.clip(img, 0.0, 1.0).astype(np.float32)
