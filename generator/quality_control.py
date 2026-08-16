"""
Automatic Quality Control
===========================
Post-generation verification that catches corrupted or invalid samples
before they enter training or evaluation.

Checks performed per sample:
  ✓ Image dimensions are exactly 1000×1000
  ✓ Images are valid (non-zero, non-constant, within [0, 255] for uint8)
  ✓ Target center is within image bounds
  ✓ Target bounding box is within search image
  ✓ Scale relationship is within expected range
  ✓ Architecture label is valid
  ✓ Metadata is complete and parseable
  ✓ Reference and search images are not identical (noise independence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class QCResult:
    """Result of quality control checks on a single sample."""

    pair_id: int
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"QC pair_{self.pair_id:03d}: {status}"]
        for e in self.errors:
            lines.append(f"  ✗ {e}")
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


def run_qc(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    metadata: Dict[str, Any],
    expected_size: int = 1000,
) -> QCResult:
    """
    Run all quality-control checks on a generated sample.

    Parameters
    ----------
    ref_img : Reference image as uint8 array.
    search_img : Search image as uint8 array.
    metadata : Per-sample metadata dict.
    expected_size : Expected image dimension.

    Returns
    -------
    QCResult with pass/fail status and error details.
    """
    pair_id = metadata.get("pair_id", -1)
    qc = QCResult(pair_id=pair_id)

    # --- Image dimension checks ---
    if ref_img.shape != (expected_size, expected_size):
        qc.fail(f"Reference image shape {ref_img.shape} != ({expected_size}, {expected_size})")
    if search_img.shape != (expected_size, expected_size):
        qc.fail(f"Search image shape {search_img.shape} != ({expected_size}, {expected_size})")

    # --- Image validity ---
    for name, img in [("Reference", ref_img), ("Search", search_img)]:
        if img.max() == 0:
            qc.fail(f"{name} image is all zeros")
        if img.min() == img.max():
            qc.fail(f"{name} image is constant (value={img.min()})")
        if img.dtype == np.uint8:
            if img.max() > 255 or img.min() < 0:
                qc.fail(f"{name} image has values outside [0, 255]")
        elif np.isnan(img).any():
            qc.fail(f"{name} image contains NaN values")

    # --- Ground-truth center within bounds ---
    cx = metadata.get("true_center_x")
    cy = metadata.get("true_center_y")
    if cx is not None and cy is not None:
        if not (0 <= cx <= expected_size):
            qc.fail(f"Target center_x={cx:.2f} outside image bounds [0, {expected_size}]")
        if not (0 <= cy <= expected_size):
            qc.fail(f"Target center_y={cy:.2f} outside image bounds [0, {expected_size}]")
    else:
        qc.fail("Metadata missing 'true_center_x' or 'true_center_y'")

    # --- Bounding box within bounds ---
    bbox = metadata.get("target_bbox", {})
    if bbox:
        for key in ("x_min", "y_min", "x_max", "y_max"):
            val = bbox.get(key)
            if val is not None:
                if val < -1.0 or val > expected_size + 1.0:
                    qc.fail(f"Bounding box {key}={val:.2f} outside image bounds")
    else:
        qc.warn("Metadata missing 'target_bbox'")

    # --- Scale relationship ---
    scale = metadata.get("scale_ratio")
    if scale is not None:
        if not (8.0 <= scale <= 12.0):
            qc.fail(f"Scale ratio {scale:.2f} outside expected range [8, 12]")
    else:
        qc.warn("Metadata missing 'scale_ratio'")

    # --- Architecture label ---
    arch = metadata.get("architecture", "")
    if arch not in ("dram", "finfet"):
        qc.fail(f"Invalid architecture label: '{arch}'")

    # --- Required metadata keys ---
    required_keys = [
        "pair_id", "architecture", "difficulty",
        "true_center_x", "true_center_y",
        "search_fov_nm", "ref_fov_nm", "seed",
    ]
    for key in required_keys:
        if key not in metadata:
            qc.fail(f"Metadata missing required key: '{key}'")

    # --- Noise independence check ---
    # If images are pixel-identical, noise was not applied independently
    if ref_img.shape == search_img.shape:
        # Only check if shapes match (they won't if either failed dim check)
        corr = np.corrcoef(ref_img.ravel().astype(np.float64),
                           search_img.ravel().astype(np.float64))[0, 1]
        if corr > 0.999:
            qc.fail(f"Reference and search images are nearly identical (corr={corr:.6f})")

    return qc
