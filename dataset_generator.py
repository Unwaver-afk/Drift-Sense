#!/usr/bin/env python3
"""
Drift-Sense Dataset Generator (v2)
====================================
Complete, modular, physics-aware synthetic SEM dataset generator for
semiconductor wafer inspection.

Features (25 deliverables):
    1.  Configurable DRAM layout renderer with randomised structural params
    2.  Configurable FinFET layout renderer with randomised structural params
    3.  High-quality reference image generation with per-sample augmentation
    4.  1000×1000 search image with realistic degradation
    5.  Multi-strategy random target placement (uniform/edge/center/hard)
    6.  Rich ground truth with bounding box + full augmentation metadata
    7.  Independent sensor noise (reference ≠ search)
    8.  SEM edge brightening with randomised strength
    9.  Variable isotropic + anisotropic blur
    10. Rotation augmentation (±2°)
    11. Scaling / magnification variation (±3%)
    12. Contrast / brightness variation
    13. Randomised structural parameters (pitches, widths, gaps)
    14. Controlled imperfections (line-width jitter, pitch jitter, missing features)
    15. Difficult periodic cases (adjustable periodicity)
    16. Multi-level difficulty (easy / medium / hard / extreme / mixed)
    17. Architecture balance (50/50 DRAM/FinFET)
    18. Comprehensive per-sample metadata (JSON)
    19. Reproducibility via per-sample random seeds
    20. Dataset splits (train / validation / hard_cases)
    21. Automatic quality control
    22. Debug visualisation mode
    23. Generator evaluation report / statistics
    24. Literature citation mapping (see citations.md)
    25. Modular package architecture (generator/)

Usage
-----
    # Basic 30-pair mixed dataset
    python dataset_generator.py --style both --num_pairs 30 --output_dir ./dataset

    # Large training set with splits and debug figures
    python dataset_generator.py --num_pairs 1000 --difficulty mixed --split --debug \\
        --output_dir ./dataset --seed 42

    # Stress test with extreme difficulty
    python dataset_generator.py --num_pairs 50 --difficulty extreme --output_dir ./stress_test
"""

import os
import sys
import json
import time
import argparse
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Local generator package
from generator.dram import DRAMConfig, render_dram_layout, random_dram_config
from generator.finfet import FinFETConfig, render_finfet_layout, random_finfet_config
from generator.noise import NoiseConfig, apply_noise, random_noise_config
from generator.edge_enhancement import EdgeConfig, apply_edge_enhancement, random_edge_config
from generator.blur import BlurConfig, apply_blur, random_blur_config
from generator.transforms import (
    TransformConfig, apply_rotation, apply_scale_jitter,
    apply_contrast_brightness, random_transform_config,
)
from generator.imperfections import ImperfectionConfig, apply_imperfections, random_imperfection_config
from generator.target_placement import PlacementConfig, compute_placement, random_placement_config
from generator.ground_truth import build_metadata, compute_bounding_box
from generator.quality_control import run_qc, QCResult
from generator.visualization import render_debug_figure, generate_dataset_stats


# ---------------------------------------------------------------------------
# Difficulty presets — pure-periodic probability
# ---------------------------------------------------------------------------
_PERIODIC_PROB = {
    "easy": 0.0,
    "medium": 0.10,
    "hard": 0.30,
    "extreme": 0.60,
}


# ---------------------------------------------------------------------------
# Core image-pair generation
# ---------------------------------------------------------------------------

def generate_image_pair(
    style: str = "dram",
    difficulty: str = "medium",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate a single (Reference, Search) SEM image pair with full augmentation.

    Parameters
    ----------
    style : "dram" or "finfet".
    difficulty : "easy", "medium", "hard", or "extreme".
    seed : Per-sample random seed for reproducibility.

    Returns
    -------
    (ref_uint8, search_uint8, metadata) where images are 1000×1000 uint8
    and metadata is a dict ready for JSON serialisation.
    """
    rng = np.random.default_rng(seed)

    # ── 1. Sample all augmentation configs for this difficulty ──────────────
    if style.lower() == "dram":
        struct_cfg = random_dram_config(rng, difficulty)
    else:
        struct_cfg = random_finfet_config(rng, difficulty)

    imperf_cfg = random_imperfection_config(rng, difficulty)
    apply_imperfections(struct_cfg, imperf_cfg)

    noise_cfg = random_noise_config(rng, difficulty)
    edge_cfg_ref = random_edge_config(rng, difficulty)
    edge_cfg_search = random_edge_config(rng, difficulty)
    blur_cfg = random_blur_config(rng, difficulty)
    transform_cfg = random_transform_config(rng, difficulty)
    placement_cfg = random_placement_config(rng, difficulty)

    # ── 2. Determine pure-periodic mode ─────────────────────────────────────
    periodic_prob = _PERIODIC_PROB.get(difficulty, 0.10)
    pure_periodic = bool(rng.random() < periodic_prob)

    # ── 3. Physical dimensions ──────────────────────────────────────────────
    search_fov_nm = 10000.0   # 10 µm
    nominal_ref_fov_nm = 1000.0   # 1 µm
    ref_fov_nm = apply_scale_jitter(nominal_ref_fov_nm, transform_cfg.scale_factor)
    search_px = 1000
    ref_px = 1000

    # Effective scale ratio
    scale_ratio = search_fov_nm / ref_fov_nm

    # ── 4. Target placement ─────────────────────────────────────────────────
    cx_nm, cy_nm = compute_placement(
        rng, placement_cfg,
        search_fov_nm=search_fov_nm,
        ref_fov_nm=ref_fov_nm,
    )

    # ── 5. Render layouts ───────────────────────────────────────────────────
    # Search canvas: 2000×2000 supersampled (5 nm/px) for anti-aliased downsample
    canvas_px = 2000
    canvas_x = np.linspace(0.0, search_fov_nm, canvas_px, endpoint=False)
    canvas_y = np.linspace(0.0, search_fov_nm, canvas_px, endpoint=False)

    # Reference rendered on its own coordinate grid
    ref_x = np.linspace(
        cx_nm - ref_fov_nm / 2.0,
        cx_nm + ref_fov_nm / 2.0,
        ref_px, endpoint=False,
    )
    ref_y = np.linspace(
        cy_nm - ref_fov_nm / 2.0,
        cy_nm + ref_fov_nm / 2.0,
        ref_px, endpoint=False,
    )

    if style.lower() == "dram":
        canvas_base = render_dram_layout(canvas_x, canvas_y, cfg=struct_cfg,
                                          pure_periodic=pure_periodic, rng=rng)
        ref_base = render_dram_layout(ref_x, ref_y, cfg=struct_cfg,
                                       pure_periodic=pure_periodic, rng=rng)
    else:
        canvas_base = render_finfet_layout(canvas_x, canvas_y, cfg=struct_cfg,
                                            pure_periodic=pure_periodic, rng=rng)
        ref_base = render_finfet_layout(ref_x, ref_y, cfg=struct_cfg,
                                          pure_periodic=pure_periodic, rng=rng)

    # ── 6. Downsample search canvas (anti-aliased area averaging) ───────────
    # Use PIL BOX resampling (pure-Python, no OpenCV dependency)
    canvas_uint8 = (canvas_base * 255.0).round().astype(np.uint8)
    search_pil = Image.fromarray(canvas_uint8).resize(
        (search_px, search_px), Image.Resampling.BOX,
    )
    search_base = np.array(search_pil, dtype=np.float32) / 255.0

    # ── 7. SEM physics pipeline (applied independently) ─────────────────────

    # --- Reference image ---
    ref_img = apply_edge_enhancement(ref_base, edge_cfg_ref)
    ref_img = apply_blur(ref_img, blur_cfg, is_search=False)
    ref_img = apply_noise(ref_img, rng, noise_cfg, is_search=False)
    ref_img = apply_contrast_brightness(
        ref_img,
        transform_cfg.contrast_factor,
        transform_cfg.brightness_offset,
    )

    # --- Search image ---
    # Search gets its own independent edge/contrast configs
    search_contrast = 1.0 + (transform_cfg.contrast_factor - 1.0) * rng.uniform(0.6, 1.4)
    search_brightness = transform_cfg.brightness_offset * rng.uniform(0.5, 1.5)

    search_img = apply_edge_enhancement(search_base, edge_cfg_search)
    search_img = apply_blur(search_img, blur_cfg, is_search=True)
    search_img = apply_noise(search_img, rng, noise_cfg, is_search=True)
    search_img = apply_contrast_brightness(search_img, search_contrast, search_brightness)

    # ── 8. Rotation (applied to reference only) ─────────────────────────────
    if abs(transform_cfg.rotation_deg) > 1e-6:
        # Use the image mean as fill value for rotation borders
        fill_val = float(np.mean(ref_img))
        ref_img = apply_rotation(ref_img, transform_cfg.rotation_deg, fill_value=fill_val)

    # ── 9. Ground truth in search-image pixel coordinates ───────────────────
    gt_x_px = (cx_nm / search_fov_nm) * search_px
    gt_y_px = (cy_nm / search_fov_nm) * search_px

    # Template size in search pixels
    template_size = int(round(ref_fov_nm / search_fov_nm * search_px))

    # ── 10. Convert to uint8 ────────────────────────────────────────────────
    ref_uint8 = (ref_img * 255.0).round().astype(np.uint8)
    search_uint8 = (search_img * 255.0).round().astype(np.uint8)

    # ── 11. Build rich metadata ─────────────────────────────────────────────
    # Extract structural params for metadata
    struct_dict = asdict(struct_cfg)

    # Compute actual noise levels applied
    noise_shot_search = noise_cfg.shot_noise_scale * noise_cfg.search_shot_factor
    noise_thermal_search = noise_cfg.thermal_sigma * noise_cfg.search_thermal_factor

    metadata = build_metadata(
        pair_id=-1,  # Will be set by the orchestrator
        style=style.lower(),
        difficulty=difficulty,
        center_x_px=gt_x_px,
        center_y_px=gt_y_px,
        center_nm_x=cx_nm,
        center_nm_y=cy_nm,
        search_fov_nm=search_fov_nm,
        ref_fov_nm=ref_fov_nm,
        scale_ratio=scale_ratio,
        pure_periodic=pure_periodic,
        seed=seed,
        rotation_deg=transform_cfg.rotation_deg,
        scale_factor=transform_cfg.scale_factor,
        noise_shot_ref=noise_cfg.shot_noise_scale,
        noise_thermal_ref=noise_cfg.thermal_sigma,
        noise_shot_search=noise_shot_search,
        noise_thermal_search=noise_thermal_search,
        blur_sigma_ref=blur_cfg.sigma,
        blur_sigma_search=blur_cfg.sigma * blur_cfg.search_sigma_factor,
        blur_anisotropy=blur_cfg.sigma_y_ratio,
        contrast_factor_ref=transform_cfg.contrast_factor,
        contrast_factor_search=search_contrast,
        brightness_offset_ref=transform_cfg.brightness_offset,
        brightness_offset_search=search_brightness,
        edge_strength_ref=edge_cfg_ref.edge_weight,
        edge_strength_search=edge_cfg_search.edge_weight,
        placement_strategy=placement_cfg.strategy,
        structural_params=struct_dict,
        template_size=template_size,
        image_size=search_px,
    )

    return ref_uint8, search_uint8, metadata


# ---------------------------------------------------------------------------
# Dataset orchestrator
# ---------------------------------------------------------------------------

def generate_dataset(
    num_pairs: int = 30,
    style: str = "both",
    difficulty: str = "mixed",
    output_dir: str = "./dataset",
    seed: int = 42,
    enable_split: bool = False,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    hard_ratio: float = 0.15,
    enable_debug: bool = False,
    enable_qc: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate a complete dataset of SEM image pairs.

    Parameters
    ----------
    num_pairs : Total number of pairs to generate.
    style : "dram", "finfet", or "both" (50/50 balanced).
    difficulty : "easy", "medium", "hard", "extreme", or "mixed".
    output_dir : Root output directory.
    seed : Master random seed.
    enable_split : If True, split into train/validation/hard_cases subdirs.
    train_ratio, val_ratio, hard_ratio : Split ratios (must sum to ≤1.0).
    enable_debug : If True, generate debug figures per sample.
    enable_qc : If True, run automatic quality control on every sample.

    Returns
    -------
    List of per-sample metadata dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    master_rng = np.random.default_rng(seed)

    # ── Architecture balance ────────────────────────────────────────────────
    styles = ["dram", "finfet"] if style == "both" else [style.lower()]

    # ── Difficulty schedule ──────────────────────────────────────────────────
    if difficulty == "mixed":
        # Balanced distribution across tiers
        diff_choices = ["easy", "medium", "hard", "extreme"]
        diff_weights = np.array([0.20, 0.35, 0.30, 0.15])
    else:
        diff_choices = [difficulty]
        diff_weights = np.array([1.0])
    diff_weights /= diff_weights.sum()

    # ── Split assignment ────────────────────────────────────────────────────
    split_labels = []
    if enable_split:
        n_train = int(round(num_pairs * train_ratio))
        n_val = int(round(num_pairs * val_ratio))
        n_hard = num_pairs - n_train - n_val
        split_labels = (["train"] * n_train + ["validation"] * n_val + ["hard_cases"] * n_hard)
        master_rng.shuffle(split_labels)
    else:
        split_labels = [None] * num_pairs

    # ── Pre-generate per-sample seeds ───────────────────────────────────────
    sample_seeds = master_rng.integers(0, 2**31, size=num_pairs).tolist()

    manifest: List[Dict[str, Any]] = []
    qc_failures = 0
    total_time = 0.0

    print(f"[*] Generating {num_pairs} image pairs ({style}, difficulty={difficulty})"
          f" into '{output_dir}'...", flush=True)
    if enable_split:
        print(f"    Splits: train={train_ratio:.0%}, val={val_ratio:.0%}, hard={hard_ratio:.0%}", flush=True)

    for i in range(num_pairs):
        t0 = time.perf_counter()

        # Architecture: alternating for balance
        sample_style = styles[i % len(styles)]

        # Difficulty
        sample_diff = str(master_rng.choice(diff_choices, p=diff_weights))

        # Split directory
        split = split_labels[i]
        if split:
            sample_dir = os.path.join(output_dir, split, f"pair_{i:03d}")
        else:
            sample_dir = os.path.join(output_dir, f"pair_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Generate
        sample_seed = sample_seeds[i]
        ref_img, search_img, meta = generate_image_pair(
            style=sample_style,
            difficulty=sample_diff,
            seed=sample_seed,
        )

        # Assign pair_id
        meta["pair_id"] = i
        meta["id"] = f"pair_{i:03d}"
        if split:
            meta["split"] = split

        # Save images
        ref_path = os.path.join(sample_dir, "reference.png")
        search_path = os.path.join(sample_dir, "search.png")
        meta_path = os.path.join(sample_dir, "ground_truth.json")

        Image.fromarray(ref_img).save(ref_path)
        Image.fromarray(search_img).save(search_path)

        # Store relative paths (portable)
        meta["reference_path"] = os.path.relpath(ref_path, output_dir)
        meta["search_path"] = os.path.relpath(search_path, output_dir)

        # Save per-sample metadata
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Quality control
        if enable_qc:
            qc_result = run_qc(ref_img, search_img, meta)
            if not qc_result.passed:
                qc_failures += 1
                print(f"    ⚠ QC FAIL pair_{i:03d}: {'; '.join(qc_result.errors)}", flush=True)

        # Debug figure
        if enable_debug:
            debug_path = os.path.join(sample_dir, "debug.png")
            render_debug_figure(ref_img, search_img, meta, debug_path)

        manifest.append(meta)

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # Progress reporting
        if (i + 1) % max(1, num_pairs // 10) == 0 or (i + 1) == num_pairs:
            avg_ms = (total_time / (i + 1)) * 1000
            print(
                f"    → {i+1}/{num_pairs}  [{sample_style:6s} {sample_diff:7s}]"
                f"  GT: ({meta['true_center_x']:.1f}, {meta['true_center_y']:.1f})"
                f"  {avg_ms:.0f} ms/pair",
                flush=True,
            )

    # ── Save manifest ───────────────────────────────────────────────────────
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Dataset statistics report ───────────────────────────────────────────
    generate_dataset_stats(manifest, output_dir)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n[+] Dataset generation complete.")
    print(f"    Manifest: {manifest_path}")
    print(f"    Total time: {total_time:.1f}s ({total_time/num_pairs*1000:.0f} ms/pair)")
    if enable_qc:
        print(f"    QC: {num_pairs - qc_failures}/{num_pairs} passed"
              f"  ({qc_failures} failures)")

    return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense Synthetic SEM Dataset Generator (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--style", type=str, default="both",
        choices=["dram", "finfet", "both"],
        help="Architecture style: dram, finfet, or both (50/50 balanced)",
    )
    parser.add_argument(
        "--num_pairs", type=int, default=30,
        help="Total number of image pairs to generate (default: 30)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dataset",
        help="Root output directory (default: ./dataset)",
    )
    parser.add_argument(
        "--difficulty", type=str, default="mixed",
        choices=["easy", "medium", "hard", "extreme", "mixed"],
        help="Difficulty level or 'mixed' for balanced distribution (default: mixed)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed for reproducibility (default: 42)",
    )

    # Split options
    parser.add_argument(
        "--split", action="store_true",
        help="Enable train/validation/hard_cases directory splits",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.70,
        help="Fraction of samples for training split (default: 0.70)",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15,
        help="Fraction of samples for validation split (default: 0.15)",
    )
    parser.add_argument(
        "--hard_ratio", type=float, default=0.15,
        help="Fraction of samples for hard_cases split (default: 0.15)",
    )

    # Debug & QC
    parser.add_argument(
        "--debug", action="store_true",
        help="Generate per-sample debug figures (pair_XXX/debug.png)",
    )
    parser.add_argument(
        "--no_qc", action="store_true",
        help="Skip automatic quality-control checks",
    )

    args = parser.parse_args()

    generate_dataset(
        num_pairs=args.num_pairs,
        style=args.style,
        difficulty=args.difficulty,
        output_dir=args.output_dir,
        seed=args.seed,
        enable_split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        hard_ratio=args.hard_ratio,
        enable_debug=args.debug,
        enable_qc=not args.no_qc,
    )


if __name__ == "__main__":
    main()
