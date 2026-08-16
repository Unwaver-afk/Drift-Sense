"""
Ground Truth & Rich Metadata Recording
========================================
Every generated sample must have a complete metadata record containing:
  * Target center (x, y) in search-image pixel coordinates
  * Target bounding box (x_min, y_min, x_max, y_max)
  * All augmentation parameters (noise, blur, rotation, scale, contrast, etc.)
  * Architecture label and difficulty tier
  * Per-sample random seed for reproducibility
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def compute_bounding_box(
    center_x_px: float,
    center_y_px: float,
    template_width: int = 100,
    template_height: int = 100,
    image_size: int = 1000,
) -> Dict[str, float]:
    """
    Compute the axis-aligned bounding box of the downsampled reference
    template within the search image.

    Parameters
    ----------
    center_x_px, center_y_px : Center of the template in search pixels.
    template_width, template_height : Size of the downsampled template (px).
    image_size : Search image dimension.

    Returns
    -------
    Dict with keys: x_min, y_min, x_max, y_max, center_x, center_y.
    """
    half_w = template_width / 2.0
    half_h = template_height / 2.0

    x_min = max(0.0, center_x_px - half_w)
    y_min = max(0.0, center_y_px - half_h)
    x_max = min(float(image_size), center_x_px + half_w)
    y_max = min(float(image_size), center_y_px + half_h)

    return {
        "x_min": round(x_min, 4),
        "y_min": round(y_min, 4),
        "x_max": round(x_max, 4),
        "y_max": round(y_max, 4),
        "center_x": round(center_x_px, 4),
        "center_y": round(center_y_px, 4),
    }


def build_metadata(
    pair_id: int,
    style: str,
    difficulty: str,
    center_x_px: float,
    center_y_px: float,
    center_nm_x: float,
    center_nm_y: float,
    search_fov_nm: float,
    ref_fov_nm: float,
    scale_ratio: float,
    pure_periodic: bool,
    seed: int,
    # Augmentation parameters
    rotation_deg: float = 0.0,
    scale_factor: float = 1.0,
    noise_shot_ref: float = 0.0,
    noise_thermal_ref: float = 0.0,
    noise_shot_search: float = 0.0,
    noise_thermal_search: float = 0.0,
    blur_sigma_ref: float = 0.0,
    blur_sigma_search: float = 0.0,
    blur_anisotropy: float = 1.0,
    contrast_factor_ref: float = 1.0,
    contrast_factor_search: float = 1.0,
    brightness_offset_ref: float = 0.0,
    brightness_offset_search: float = 0.0,
    edge_strength_ref: float = 0.0,
    edge_strength_search: float = 0.0,
    placement_strategy: str = "uniform",
    structural_params: Optional[Dict[str, Any]] = None,
    template_size: int = 100,
    image_size: int = 1000,
) -> Dict[str, Any]:
    """
    Build a complete metadata record for a single generated pair.

    Returns a dictionary ready for JSON serialisation.
    """
    bbox = compute_bounding_box(
        center_x_px, center_y_px,
        template_width=template_size,
        template_height=template_size,
        image_size=image_size,
    )

    return {
        "id": f"pair_{pair_id:03d}",
        "pair_id": pair_id,
        "architecture": style,
        "difficulty": difficulty,

        # Ground truth
        "true_center_x": round(center_x_px, 4),
        "true_center_y": round(center_y_px, 4),
        "target_bbox": bbox,
        "center_nm_x": round(center_nm_x, 4),
        "center_nm_y": round(center_nm_y, 4),

        # Image specs
        "reference_size": [image_size, image_size],
        "search_size": [image_size, image_size],
        "search_fov_nm": search_fov_nm,
        "ref_fov_nm": round(ref_fov_nm, 4),
        "scale_ratio": round(scale_ratio, 4),
        "template_size": template_size,

        # Transforms
        "rotation_deg": round(rotation_deg, 4),
        "scale_factor": round(scale_factor, 4),

        # Noise
        "noise_shot_ref": round(noise_shot_ref, 6),
        "noise_thermal_ref": round(noise_thermal_ref, 6),
        "noise_shot_search": round(noise_shot_search, 6),
        "noise_thermal_search": round(noise_thermal_search, 6),

        # Blur
        "blur_sigma_ref": round(blur_sigma_ref, 4),
        "blur_sigma_search": round(blur_sigma_search, 4),
        "blur_anisotropy": round(blur_anisotropy, 4),

        # Contrast
        "contrast_factor_ref": round(contrast_factor_ref, 4),
        "contrast_factor_search": round(contrast_factor_search, 4),
        "brightness_offset_ref": round(brightness_offset_ref, 4),
        "brightness_offset_search": round(brightness_offset_search, 4),

        # Edge enhancement
        "edge_strength_ref": round(edge_strength_ref, 4),
        "edge_strength_search": round(edge_strength_search, 4),

        # Placement
        "placement_strategy": placement_strategy,
        "pure_periodic": pure_periodic,

        # Structural
        "structural_params": structural_params or {},

        # Reproducibility
        "seed": seed,
    }
