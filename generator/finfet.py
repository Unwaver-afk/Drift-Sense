"""
Configurable FinFET Standard-Cell Logic Layout Renderer
=======================================================
Generates physically-realistic FinFET die patterns with parameterized
fin pitch, gate pitch, standard-cell boundaries, poly gate cuts,
power rails, and optional controlled imperfections.

Structural references:
    Auth et al. (2012) — 22 nm Tri-Gate CMOS, IEEE VLSI Symp.
    Typical advanced FinFET: 20–36 nm fin pitch, 70–130 nm gate pitch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FinFETConfig:
    """All tuneable parameters for a single FinFET layout render."""

    # Core array pitches & widths (nm)
    fin_pitch: float = 28.0
    fin_width: float = 8.0
    gate_pitch: float = 100.0
    gate_width: float = 22.0

    # Standard-cell boundaries (nm)
    cell_height: float = 1200.0
    cell_width: float = 1680.0
    cell_boundary_width_y: float = 90.0
    cell_boundary_width_x: float = 84.0

    # Gate cuts
    gate_cut_width: float = 84.0
    gate_cut_period: float = 560.0

    # Intensities (0-1 float, SEM grayscale)
    background_intensity: float = 0.12
    fin_intensity: float = 0.35
    gate_intensity: float = 0.38
    power_rail_intensity: float = 0.50
    macro_boundary_intensity: float = 0.35

    # Macro boundary (nm from edge of 10 µm FOV)
    macro_boundary_inset: float = 800.0

    # Imperfection budget
    line_width_jitter: float = 0.0
    pitch_jitter: float = 0.0
    missing_feature_prob: float = 0.0
    intensity_variation: float = 0.0


def random_finfet_config(
    rng: np.random.Generator,
    difficulty: str = "medium",
) -> FinFETConfig:
    """Sample a randomised FinFET configuration for the given difficulty tier."""

    fin_p = rng.uniform(20.0, 36.0)
    fin_w = rng.uniform(5.0, min(12.0, fin_p * 0.45))
    gate_p = rng.uniform(70.0, 130.0)
    gate_w = rng.uniform(14.0, min(30.0, gate_p * 0.35))

    cell_h = rng.uniform(900.0, 1500.0)
    cell_w = rng.uniform(1400.0, 2000.0)
    cb_wy = rng.uniform(60.0, 120.0)
    cb_wx = rng.uniform(56.0, 112.0)
    gc_w = rng.uniform(60.0, 100.0)
    gc_period = rng.uniform(400.0, 700.0)

    bg = rng.uniform(0.08, 0.18)
    feat_scale = rng.uniform(0.85, 1.15)

    jitter_map = {"easy": 0.0, "medium": 0.03, "hard": 0.08, "extreme": 0.15}
    missing_map = {"easy": 0.0, "medium": 0.005, "hard": 0.01, "extreme": 0.02}
    intvar_map = {"easy": 0.0, "medium": 0.03, "hard": 0.06, "extreme": 0.10}

    j = jitter_map.get(difficulty, 0.03)
    m = missing_map.get(difficulty, 0.005)
    iv = intvar_map.get(difficulty, 0.03)

    return FinFETConfig(
        fin_pitch=fin_p,
        fin_width=fin_w,
        gate_pitch=gate_p,
        gate_width=gate_w,
        cell_height=cell_h,
        cell_width=cell_w,
        cell_boundary_width_y=cb_wy,
        cell_boundary_width_x=cb_wx,
        gate_cut_width=gc_w,
        gate_cut_period=gc_period,
        background_intensity=bg,
        fin_intensity=0.35 * feat_scale,
        gate_intensity=0.38 * feat_scale,
        power_rail_intensity=0.50 * feat_scale,
        macro_boundary_intensity=0.35 * feat_scale,
        line_width_jitter=j,
        pitch_jitter=j,
        missing_feature_prob=m,
        intensity_variation=iv,
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_finfet_layout(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    cfg: Optional[FinFETConfig] = None,
    pure_periodic: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Render a FinFET layout image over the given physical coordinate arrays.

    Parameters
    ----------
    x_coords, y_coords : 1-D arrays of physical positions (nm).
    cfg : FinFETConfig — if None, defaults are used.
    pure_periodic : If True, omit all macro-structures (infinite periodic grid).
    rng : Random generator used for imperfections.

    Returns
    -------
    2-D float32 array in [0, 1] with shape (len(y_coords), len(x_coords)).
    """
    if cfg is None:
        cfg = FinFETConfig()
    if rng is None:
        rng = np.random.default_rng()

    X, Y = np.meshgrid(x_coords, y_coords)
    H, W = X.shape

    # --- Pitch jitter --------------------------------------------------------
    if cfg.pitch_jitter > 0:
        from .dram import _smooth_noise

        jitter_fx = _smooth_noise(H, W, rng, scale=32) * cfg.pitch_jitter * cfg.fin_pitch
        jitter_fy = _smooth_noise(H, W, rng, scale=32) * cfg.pitch_jitter * cfg.gate_pitch
        X_j = X + jitter_fx
        Y_j = Y + jitter_fy
    else:
        X_j, Y_j = X, Y

    # --- Core periodic array -------------------------------------------------
    fp, fw = cfg.fin_pitch, cfg.fin_width
    gp, gw = cfg.gate_pitch, cfg.gate_width

    dist_fin = np.abs((X_j % fp) - (fp / 2.0))
    dist_gate = np.abs((Y_j % gp) - (gp / 2.0))

    # Line-width jitter
    fin_hw = fw / 2.0
    gate_hw = gw / 2.0
    if cfg.line_width_jitter > 0:
        from .dram import _smooth_noise

        fin_hw_f = fin_hw * (1.0 + cfg.line_width_jitter * _smooth_noise(H, W, rng, scale=16))
        gate_hw_f = gate_hw * (1.0 + cfg.line_width_jitter * _smooth_noise(H, W, rng, scale=16))
    else:
        fin_hw_f = fin_hw
        gate_hw_f = gate_hw

    fins = (dist_fin <= fin_hw_f).astype(np.float32)
    gates = (dist_gate <= gate_hw_f).astype(np.float32)

    if pure_periodic:
        base = (cfg.background_intensity
                + cfg.fin_intensity * fins
                + cfg.gate_intensity * gates)
        if cfg.intensity_variation > 0:
            from .dram import _smooth_noise
            base *= (1.0 + cfg.intensity_variation * _smooth_noise(H, W, rng, scale=64))
        return np.clip(base, 0.0, 1.0).astype(np.float32)

    # --- Macro-structures ----------------------------------------------------
    ch, cw = cfg.cell_height, cfg.cell_width
    cb_y = cfg.cell_boundary_width_y
    cb_x = cfg.cell_boundary_width_x

    cell_boundary_y_bool = (Y % ch) < cb_y
    cell_boundary_x_bool = (X % cw) < cb_x

    # Gate cuts
    gate_cut_y = np.abs((Y % ch) - (ch / 2.0)) < (gw * 0.95)
    gate_cut_x = (X % cfg.gate_cut_period) < cfg.gate_cut_width
    gate_cuts = (gate_cut_y & gate_cut_x).astype(np.float32)

    # Missing gate features
    if cfg.missing_feature_prob > 0:
        from .dram import _drop_features
        gate_cuts_extra = _drop_features(gates, cfg.missing_feature_prob, rng)
        # Use the dropped version for active gates
        active_gates_raw = gate_cuts_extra
    else:
        active_gates_raw = gates

    active_gates = np.clip(
        active_gates_raw * (~cell_boundary_y_bool).astype(np.float32) - gate_cuts,
        0.0, 1.0,
    )
    active_fins = np.clip(
        fins * (~cell_boundary_x_bool).astype(np.float32),
        0.0, 1.0,
    )

    power_rails = cell_boundary_y_bool.astype(np.float32)

    # Macro boundary — only meaningful for the full 10 µm search canvas
    fov_extent_x = x_coords[-1] - x_coords[0]
    fov_extent_y = y_coords[-1] - y_coords[0]
    inset = cfg.macro_boundary_inset
    if fov_extent_x > 5000.0:  # only for search-scale FOV
        macro_boundary = (
            (X < (x_coords[0] + inset))
            | (X > (x_coords[-1] - inset))
            | (Y < (y_coords[0] + inset))
            | (Y > (y_coords[-1] - inset))
        ).astype(np.float32)
    else:
        macro_boundary = np.zeros_like(X, dtype=np.float32)

    base = (cfg.background_intensity
            + cfg.fin_intensity * active_fins
            + cfg.gate_intensity * active_gates
            + cfg.power_rail_intensity * power_rails
            + cfg.macro_boundary_intensity * macro_boundary)

    if cfg.intensity_variation > 0:
        from .dram import _smooth_noise
        base *= (1.0 + cfg.intensity_variation * _smooth_noise(H, W, rng, scale=64))

    return np.clip(base, 0.0, 1.0).astype(np.float32)
