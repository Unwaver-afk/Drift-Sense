"""
Drift-Sense Dataset Generator Package
======================================
Modular, physics-aware synthetic SEM dataset generator for semiconductor
wafer inspection (DRAM & FinFET layouts).

Submodules
----------
dram              Configurable DRAM memory-array renderer
finfet            Configurable FinFET standard-cell renderer
noise             Multi-model independent SEM sensor noise
edge_enhancement  Secondary-electron edge-brightening simulation
blur              Isotropic / anisotropic / PSF-inspired blur
transforms        Rotation, scale jitter, contrast & brightness
imperfections     Controlled structural imperfections
target_placement  Multi-strategy target placement within search FOV
ground_truth      Rich per-sample metadata & bounding-box recording
quality_control   Automatic post-generation verification
visualization     Debug figures & dataset statistics report
"""

from .dram import DRAMConfig, render_dram_layout, random_dram_config
from .finfet import FinFETConfig, render_finfet_layout, random_finfet_config
from .noise import apply_noise, NoiseConfig, random_noise_config
from .edge_enhancement import apply_edge_enhancement, EdgeConfig, random_edge_config
from .blur import apply_blur, BlurConfig, random_blur_config
from .transforms import (
    apply_rotation, apply_scale_jitter, apply_contrast_brightness,
    TransformConfig, random_transform_config,
)
from .imperfections import apply_imperfections, ImperfectionConfig, random_imperfection_config
from .target_placement import compute_placement, PlacementConfig, random_placement_config
from .ground_truth import build_metadata, compute_bounding_box
from .quality_control import run_qc, QCResult
from .visualization import render_debug_figure, generate_dataset_stats

__all__ = [
    # DRAM
    "DRAMConfig", "render_dram_layout", "random_dram_config",
    # FinFET
    "FinFETConfig", "render_finfet_layout", "random_finfet_config",
    # Noise
    "apply_noise", "NoiseConfig", "random_noise_config",
    # Edge
    "apply_edge_enhancement", "EdgeConfig", "random_edge_config",
    # Blur
    "apply_blur", "BlurConfig", "random_blur_config",
    # Transforms
    "apply_rotation", "apply_scale_jitter", "apply_contrast_brightness",
    "TransformConfig", "random_transform_config",
    # Imperfections
    "apply_imperfections", "ImperfectionConfig", "random_imperfection_config",
    # Placement
    "compute_placement", "PlacementConfig", "random_placement_config",
    # Ground truth
    "build_metadata", "compute_bounding_box",
    # QC
    "run_qc", "QCResult",
    # Visualization
    "render_debug_figure", "generate_dataset_stats",
]
