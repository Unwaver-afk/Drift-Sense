"""
Visualization & Dataset Statistics
=====================================
Two capabilities:

1. **Per-sample debug figure** — Reference + Search with GT bounding box,
   annotated with augmentation parameters.
2. **Dataset statistics report** — Aggregate summary of all samples
   (architecture split, noise/blur/rotation/scale distributions).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

# matplotlib import is deferred to avoid loading it unless needed
_MPL_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    _MPL_AVAILABLE = False


def render_debug_figure(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    metadata: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Render a 2-panel debug figure for a single sample.

    Panel 1: Reference image
    Panel 2: Search image with GT bounding box + center marker + annotations

    Parameters
    ----------
    ref_img : Reference image (uint8 or float32).
    search_img : Search image (uint8 or float32).
    metadata : Per-sample metadata dict.
    output_path : File path to save the figure (e.g. "pair_000/debug.png").
    """
    if not _MPL_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    style = metadata.get("architecture", "?").upper()
    difficulty = metadata.get("difficulty", "?")
    pair_id = metadata.get("pair_id", -1)

    # Panel 1: Reference
    axes[0].imshow(ref_img, cmap="gray", vmin=0, vmax=255 if ref_img.dtype == np.uint8 else 1.0)
    axes[0].set_title(
        f"Reference (1 µm × 1 µm, 1 nm/px)\n{style} | Difficulty: {difficulty}",
        color="#38bdf8", fontsize=11, pad=8,
    )
    axes[0].axis("off")

    # Panel 2: Search with GT
    axes[1].imshow(search_img, cmap="gray", vmin=0, vmax=255 if search_img.dtype == np.uint8 else 1.0)

    cx = metadata.get("true_center_x", 500)
    cy = metadata.get("true_center_y", 500)
    bbox = metadata.get("target_bbox", {})

    if bbox:
        x_min = bbox.get("x_min", cx - 50)
        y_min = bbox.get("y_min", cy - 50)
        w = bbox.get("x_max", cx + 50) - x_min
        h = bbox.get("y_max", cy + 50) - y_min
        rect = patches.Rectangle(
            (x_min, y_min), w, h,
            linewidth=2, edgecolor="#22c55e", facecolor="none", label="GT BBox",
        )
        axes[1].add_patch(rect)

    axes[1].plot(cx, cy, "g+", markersize=12, markeredgewidth=2, label="GT Center")

    # Annotation text
    rot = metadata.get("rotation_deg", 0)
    sc = metadata.get("scale_factor", 1.0)
    noise_s = metadata.get("noise_shot_search", 0)
    blur_s = metadata.get("blur_sigma_search", 0)
    contrast_s = metadata.get("contrast_factor_search", 1.0)
    placement = metadata.get("placement_strategy", "?")
    periodic = "Yes" if metadata.get("pure_periodic", False) else "No"

    annot = (
        f"GT: ({cx:.1f}, {cy:.1f})\n"
        f"Rot: {rot:.2f}° | Scale: {sc:.3f}\n"
        f"Noise: {noise_s:.3f} | Blur: {blur_s:.2f}\n"
        f"Contrast: {contrast_s:.2f} | Place: {placement}\n"
        f"Periodic: {periodic}"
    )
    axes[1].text(
        10, 30, annot,
        color="#e2e8f0", fontsize=8,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor="#334155", alpha=0.85),
    )
    axes[1].set_title(
        f"Search (10 µm × 10 µm, 10 nm/px) — pair_{pair_id:03d}",
        color="#38bdf8", fontsize=11, pad=8,
    )
    axes[1].legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    axes[1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def generate_dataset_stats(
    manifest: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Generate aggregate dataset statistics and save to ``dataset_stats.json``.

    Also prints a human-readable summary to stdout.

    Parameters
    ----------
    manifest : List of per-sample metadata dicts.
    output_dir : Directory to save ``dataset_stats.json``.

    Returns
    -------
    Statistics dict.
    """
    n = len(manifest)
    if n == 0:
        return {}

    # Architecture split
    dram_count = sum(1 for m in manifest if m.get("architecture") == "dram")
    finfet_count = n - dram_count

    # Difficulty distribution
    difficulties = {}
    for m in manifest:
        d = m.get("difficulty", "unknown")
        difficulties[d] = difficulties.get(d, 0) + 1

    # Placement distribution
    placements = {}
    for m in manifest:
        p = m.get("placement_strategy", "unknown")
        placements[p] = placements.get(p, 0) + 1

    # Numeric stats
    def _stats(key: str) -> Dict[str, float]:
        vals = [m.get(key, 0) for m in manifest if m.get(key) is not None]
        if not vals:
            return {"min": 0, "max": 0, "mean": 0, "std": 0}
        arr = np.array(vals, dtype=np.float64)
        return {
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
        }

    stats = {
        "total_samples": n,
        "architecture": {"dram": dram_count, "finfet": finfet_count},
        "difficulty_distribution": difficulties,
        "placement_distribution": placements,
        "noise_shot_ref": _stats("noise_shot_ref"),
        "noise_shot_search": _stats("noise_shot_search"),
        "noise_thermal_ref": _stats("noise_thermal_ref"),
        "noise_thermal_search": _stats("noise_thermal_search"),
        "blur_sigma_ref": _stats("blur_sigma_ref"),
        "blur_sigma_search": _stats("blur_sigma_search"),
        "rotation_deg": _stats("rotation_deg"),
        "scale_factor": _stats("scale_factor"),
        "contrast_factor_ref": _stats("contrast_factor_ref"),
        "contrast_factor_search": _stats("contrast_factor_search"),
        "edge_strength_ref": _stats("edge_strength_ref"),
        "edge_strength_search": _stats("edge_strength_search"),
        "pure_periodic_count": sum(1 for m in manifest if m.get("pure_periodic", False)),
    }

    # Print summary
    print("\n" + "=" * 56)
    print("           DATASET GENERATION STATISTICS")
    print("=" * 56)
    print(f"  Total samples:          {n}")
    print(f"  DRAM:                   {dram_count}")
    print(f"  FinFET:                 {finfet_count}")
    print(f"  Pure periodic:          {stats['pure_periodic_count']}")
    print()
    print("  Difficulty distribution:")
    for d, c in sorted(difficulties.items()):
        print(f"    {d:12s}  {c:4d}")
    print()
    print("  Placement distribution:")
    for p, c in sorted(placements.items()):
        print(f"    {p:12s}  {c:4d}")
    print()
    s = stats
    print(f"  Noise (shot, ref):      {s['noise_shot_ref']['min']:.4f} – {s['noise_shot_ref']['max']:.4f}  (mean {s['noise_shot_ref']['mean']:.4f})")
    print(f"  Noise (shot, search):   {s['noise_shot_search']['min']:.4f} – {s['noise_shot_search']['max']:.4f}  (mean {s['noise_shot_search']['mean']:.4f})")
    print(f"  Blur σ (ref):           {s['blur_sigma_ref']['min']:.2f} – {s['blur_sigma_ref']['max']:.2f}")
    print(f"  Blur σ (search):        {s['blur_sigma_search']['min']:.2f} – {s['blur_sigma_search']['max']:.2f}")
    print(f"  Rotation:               {s['rotation_deg']['min']:.2f}° – {s['rotation_deg']['max']:.2f}°")
    print(f"  Scale factor:           {s['scale_factor']['min']:.4f} – {s['scale_factor']['max']:.4f}")
    print(f"  Contrast (ref):         {s['contrast_factor_ref']['min']:.2f} – {s['contrast_factor_ref']['max']:.2f}")
    print(f"  Edge strength (ref):    {s['edge_strength_ref']['min']:.2f} – {s['edge_strength_ref']['max']:.2f}")
    print("=" * 56 + "\n")

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[+] Dataset statistics saved to: {stats_path}")

    return stats
