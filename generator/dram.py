"""
Configurable DRAM Memory-Array Layout Renderer
===============================================
Generates physically-realistic DRAM die patterns with parameterized
structural dimensions (bitline/wordline pitch, line width, via radius,
sense-amp gaps, power mesh) and optional controlled imperfections.

Structural references:
    IRDS 2022/2023 — IEEE IRDS Metrology & Memory Working Group
    Typical advanced DRAM: 30–50 nm bitline pitch, 45–75 nm wordline pitch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DRAMConfig:
    """All tuneable parameters for a single DRAM layout render."""

    # Core array pitches & widths (nm)
    bitline_pitch: float = 40.0
    wordline_pitch: float = 60.0
    bitline_width: float = 6.0
    wordline_width: float = 8.0
    via_radius: float = 8.0

    # Hierarchical macro-structures (nm)
    sense_amp_gap: float = 220.0
    sense_amp_period: float = 2500.0
    wl_driver_gap: float = 260.0
    wl_driver_period: float = 3200.0
    power_rail_width: float = 100.0
    global_trunk_width: float = 140.0

    # Intensities (0-1 float, SEM grayscale)
    background_intensity: float = 0.15
    bitline_intensity: float = 0.35
    wordline_intensity: float = 0.35
    via_intensity: float = 0.15
    power_rail_intensity: float = 0.30
    global_trunk_intensity: float = 0.40

    # Imperfection budget (fractions, applied externally)
    line_width_jitter: float = 0.0       # fraction of nominal width
    pitch_jitter: float = 0.0            # fraction of nominal pitch
    missing_feature_prob: float = 0.0    # probability per via of being absent
    intensity_variation: float = 0.0     # max ± relative intensity modulation


def random_dram_config(
    rng: np.random.Generator,
    difficulty: str = "medium",
) -> DRAMConfig:
    """Sample a randomised DRAM configuration for the given difficulty tier."""

    # Base structural ranges (physically plausible)
    bl_pitch = rng.uniform(30.0, 50.0)
    wl_pitch = rng.uniform(45.0, 75.0)
    bl_width = rng.uniform(4.0, 0.20 * bl_pitch)   # ≤20 % of pitch
    wl_width = rng.uniform(5.0, 0.20 * wl_pitch)
    via_r = rng.uniform(max(bl_width, 5.0), min(bl_pitch, wl_pitch) * 0.25)

    # Macro-structure variation
    sa_gap = rng.uniform(180.0, 280.0)
    sa_period = rng.uniform(2000.0, 3000.0)
    wl_drv_gap = rng.uniform(200.0, 320.0)
    wl_drv_period = rng.uniform(2800.0, 3600.0)

    # Intensity variation
    bg = rng.uniform(0.08, 0.22)
    feat_scale = rng.uniform(0.85, 1.15)

    # Imperfections scale with difficulty
    jitter_map = {"easy": 0.0, "medium": 0.03, "hard": 0.08, "extreme": 0.15}
    missing_map = {"easy": 0.0, "medium": 0.005, "hard": 0.01, "extreme": 0.02}
    intvar_map = {"easy": 0.0, "medium": 0.03, "hard": 0.06, "extreme": 0.10}

    j = jitter_map.get(difficulty, 0.03)
    m = missing_map.get(difficulty, 0.005)
    iv = intvar_map.get(difficulty, 0.03)

    return DRAMConfig(
        bitline_pitch=bl_pitch,
        wordline_pitch=wl_pitch,
        bitline_width=bl_width,
        wordline_width=wl_width,
        via_radius=via_r,
        sense_amp_gap=sa_gap,
        sense_amp_period=sa_period,
        wl_driver_gap=wl_drv_gap,
        wl_driver_period=wl_drv_period,
        background_intensity=bg,
        bitline_intensity=0.35 * feat_scale,
        wordline_intensity=0.35 * feat_scale,
        via_intensity=0.15 * feat_scale,
        power_rail_intensity=0.30 * feat_scale,
        global_trunk_intensity=0.40 * feat_scale,
        line_width_jitter=j,
        pitch_jitter=j,
        missing_feature_prob=m,
        intensity_variation=iv,
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_dram_layout(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    cfg: Optional[DRAMConfig] = None,
    pure_periodic: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Render a DRAM layout image over the given physical coordinate arrays.

    Parameters
    ----------
    x_coords, y_coords : 1-D arrays of physical positions (nm).
    cfg : DRAMConfig — if None, defaults are used.
    pure_periodic : If True, omit all macro-structures (infinite periodic grid).
    rng : Random generator used for imperfections.  Ignored when jitter is 0.

    Returns
    -------
    2-D float32 array in [0, 1] with shape (len(y_coords), len(x_coords)).
    """
    if cfg is None:
        cfg = DRAMConfig()
    if rng is None:
        rng = np.random.default_rng()

    X, Y = np.meshgrid(x_coords, y_coords)
    H, W = X.shape

    # --- Pitch jitter (spatially-smooth random offsets) ----------------------
    if cfg.pitch_jitter > 0:
        # Low-frequency spatial modulation field (smooth noise)
        jitter_field_x = _smooth_noise(H, W, rng, scale=32) * cfg.pitch_jitter * cfg.bitline_pitch
        jitter_field_y = _smooth_noise(H, W, rng, scale=32) * cfg.pitch_jitter * cfg.wordline_pitch
        X_j = X + jitter_field_x
        Y_j = Y + jitter_field_y
    else:
        X_j, Y_j = X, Y

    # --- Core periodic array -------------------------------------------------
    px, py = cfg.bitline_pitch, cfg.wordline_pitch

    dist_x = np.abs((X_j % px) - (px / 2.0))
    dist_y = np.abs((Y_j % py) - (py / 2.0))

    # Line-width jitter
    bl_hw = cfg.bitline_width / 2.0
    wl_hw = cfg.wordline_width / 2.0
    if cfg.line_width_jitter > 0:
        bl_hw_field = bl_hw * (1.0 + cfg.line_width_jitter * _smooth_noise(H, W, rng, scale=16))
        wl_hw_field = wl_hw * (1.0 + cfg.line_width_jitter * _smooth_noise(H, W, rng, scale=16))
    else:
        bl_hw_field = bl_hw
        wl_hw_field = wl_hw

    bit_lines = (dist_x <= bl_hw_field).astype(np.float32)
    word_lines = (dist_y <= wl_hw_field).astype(np.float32)

    via_dist_sq = dist_x ** 2 + dist_y ** 2
    vias = (via_dist_sq <= cfg.via_radius ** 2).astype(np.float32)

    # Missing features
    if cfg.missing_feature_prob > 0:
        vias = _drop_features(vias, cfg.missing_feature_prob, rng)

    if pure_periodic:
        base = (cfg.background_intensity
                + cfg.bitline_intensity * bit_lines
                + cfg.wordline_intensity * word_lines
                + cfg.via_intensity * vias)
        if cfg.intensity_variation > 0:
            base *= (1.0 + cfg.intensity_variation * _smooth_noise(H, W, rng, scale=64))
        return np.clip(base, 0.0, 1.0).astype(np.float32)

    # --- Hierarchical macro-structures ---------------------------------------
    sa_p = cfg.sense_amp_period
    sa_g = cfg.sense_amp_gap
    wld_p = cfg.wl_driver_period
    wld_g = cfg.wl_driver_gap

    sense_amp_gap_bool = (Y % sa_p) < sa_g
    wl_driver_gap_bool = (X % wld_p) < wld_g

    active_array = (~sense_amp_gap_bool & ~wl_driver_gap_bool).astype(np.float32)

    pr_w = cfg.power_rail_width
    power_rails = (
        (((Y % sa_p) >= (sa_g * 0.27)) & ((Y % sa_p) <= (sa_g * 0.27 + pr_w)))
        | (((X % wld_p) >= (wld_g * 0.31)) & ((X % wld_p) <= (wld_g * 0.31 + pr_w)))
    ).astype(np.float32)

    fov_center_x = (x_coords[0] + x_coords[-1]) / 2.0
    fov_center_y = (y_coords[0] + y_coords[-1]) / 2.0
    gt_w = cfg.global_trunk_width
    global_trunk_y = (np.abs(Y - fov_center_y) < gt_w).astype(np.float32)
    global_trunk_x = (np.abs(X - fov_center_x) < gt_w).astype(np.float32)

    base = (cfg.background_intensity
            + cfg.bitline_intensity * (bit_lines * active_array)
            + cfg.wordline_intensity * (word_lines * active_array)
            + cfg.via_intensity * (vias * active_array)
            + cfg.power_rail_intensity * power_rails
            + cfg.global_trunk_intensity * (global_trunk_x + global_trunk_y))

    if cfg.intensity_variation > 0:
        base *= (1.0 + cfg.intensity_variation * _smooth_noise(H, W, rng, scale=64))

    return np.clip(base, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth_noise(H: int, W: int, rng: np.random.Generator, scale: int = 32) -> np.ndarray:
    """Generate a spatially-smooth noise field in [-1, 1] via low-res upsampling."""
    from scipy.ndimage import zoom

    small_h = max(2, H // scale)
    small_w = max(2, W // scale)
    small = rng.uniform(-1.0, 1.0, (small_h, small_w)).astype(np.float32)
    zoomed = zoom(small, (H / small_h, W / small_w), order=1)
    # Ensure exact output shape (zoom can be off-by-one)
    return zoomed[:H, :W]


def _drop_features(
    feature_map: np.ndarray,
    prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly zero-out connected blobs in a binary feature map."""
    if prob <= 0:
        return feature_map
    # For efficiency, drop at downsampled grid then upsample the mask
    from scipy.ndimage import label, zoom

    ds = 4
    H, W = feature_map.shape
    small = feature_map[::ds, ::ds]
    labeled, n_features = label(small > 0.5)
    if n_features == 0:
        return feature_map
    drop_ids = set()
    for fid in range(1, n_features + 1):
        if rng.random() < prob:
            drop_ids.add(fid)
    if not drop_ids:
        return feature_map
    drop_mask_small = np.isin(labeled, list(drop_ids))
    drop_mask = zoom(drop_mask_small.astype(np.float32), (H / drop_mask_small.shape[0], W / drop_mask_small.shape[1]), order=0)
    drop_mask = drop_mask[:H, :W] > 0.5
    result = feature_map.copy()
    result[drop_mask] = 0.0
    return result
