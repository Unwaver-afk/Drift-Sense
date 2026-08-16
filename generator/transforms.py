"""
Geometric & Photometric Transforms
====================================
Implements rotation, scale jitter, and contrast/brightness variation
applied to SEM images to simulate real-world acquisition variation.

* **Rotation**: Simulates stage alignment error (±2° typical)
* **Scale jitter**: Simulates magnification calibration drift (±3% around 10×)
* **Contrast/Brightness**: Simulates detector gain and beam-current variation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage


@dataclass
class TransformConfig:
    """Parameters controlling geometric and photometric transforms."""

    # Rotation (degrees)
    rotation_deg: float = 0.0
    max_rotation_deg: float = 2.0  # Used only by random_transform_config

    # Scale jitter (fraction around 1.0)
    scale_factor: float = 1.0
    max_scale_jitter: float = 0.03  # ±3% of nominal

    # Contrast & brightness
    contrast_factor: float = 1.0    # Multiplicative
    brightness_offset: float = 0.0  # Additive
    contrast_range: Tuple[float, float] = (0.85, 1.15)
    brightness_range: Tuple[float, float] = (-0.05, 0.05)


def random_transform_config(
    rng: np.random.Generator,
    difficulty: str = "medium",
) -> TransformConfig:
    """Sample randomised transform parameters for the given difficulty."""
    rot_map = {
        "easy":    0.0,
        "medium":  0.5,
        "hard":    1.0,
        "extreme": 2.0,
    }
    scale_map = {
        "easy":    0.0,
        "medium":  0.01,
        "hard":    0.02,
        "extreme": 0.03,
    }
    contrast_map = {
        "easy":    (0.95, 1.05),
        "medium":  (0.85, 1.15),
        "hard":    (0.75, 1.25),
        "extreme": (0.65, 1.35),
    }
    brightness_map = {
        "easy":    (-0.02, 0.02),
        "medium":  (-0.05, 0.05),
        "hard":    (-0.08, 0.08),
        "extreme": (-0.10, 0.10),
    }

    max_rot = rot_map.get(difficulty, 0.5)
    max_scale = scale_map.get(difficulty, 0.01)
    c_range = contrast_map.get(difficulty, (0.85, 1.15))
    b_range = brightness_map.get(difficulty, (-0.05, 0.05))

    return TransformConfig(
        rotation_deg=rng.uniform(-max_rot, max_rot) if max_rot > 0 else 0.0,
        max_rotation_deg=max_rot,
        scale_factor=1.0 + rng.uniform(-max_scale, max_scale) if max_scale > 0 else 1.0,
        max_scale_jitter=max_scale,
        contrast_factor=rng.uniform(*c_range),
        brightness_offset=rng.uniform(*b_range),
        contrast_range=c_range,
        brightness_range=b_range,
    )


def apply_rotation(
    image: np.ndarray,
    angle_deg: float,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Rotate an image by the given angle (degrees, counter-clockwise).

    Uses bilinear interpolation with constant-value boundary fill.

    Parameters
    ----------
    image : 2-D float32 array.
    angle_deg : Rotation angle in degrees.
    fill_value : Value for pixels outside the original bounds.

    Returns
    -------
    Rotated image, same shape as input, float32.
    """
    if abs(angle_deg) < 1e-6:
        return image.copy()

    rotated = ndimage.rotate(
        image.astype(np.float32),
        angle_deg,
        reshape=False,       # keep same dimensions
        order=1,             # bilinear
        mode="constant",
        cval=fill_value,
    )
    return np.clip(rotated, 0.0, 1.0).astype(np.float32)


def apply_scale_jitter(
    ref_fov_nm: float,
    scale_factor: float,
) -> float:
    """
    Apply scale jitter to the reference FOV.

    Instead of exactly 1000 nm (1 µm), the reference FOV becomes
    1000 * scale_factor nm, simulating magnification calibration drift.

    Parameters
    ----------
    ref_fov_nm : Nominal reference FOV in nm (typically 1000.0).
    scale_factor : Jitter factor (e.g. 1.02 means 2% larger FOV).

    Returns
    -------
    Adjusted reference FOV in nm.
    """
    return ref_fov_nm * scale_factor


def apply_contrast_brightness(
    image: np.ndarray,
    contrast_factor: float = 1.0,
    brightness_offset: float = 0.0,
) -> np.ndarray:
    """
    Apply contrast scaling and brightness offset.

    new_pixel = pixel * contrast_factor + brightness_offset

    Parameters
    ----------
    image : 2-D float32 array in [0, 1].
    contrast_factor : Multiplicative contrast scaling.
    brightness_offset : Additive brightness shift.

    Returns
    -------
    Adjusted image, float32, clipped to [0, 1].
    """
    result = image.astype(np.float32) * contrast_factor + brightness_offset
    return np.clip(result, 0.0, 1.0).astype(np.float32)
