#!/usr/bin/env python3
"""
Drift-Sense Dataset Generator
=============================
Generates physically realistic synthetic SEM (Scanning Electron Microscope) image pairs:
1. High-resolution Reference Image (1000x1000 px, 1 nm/px, 1 µm x 1 µm FOV)
2. Wide Search Image (1000x1000 px, 10 nm/px, 10 µm x 10 µm FOV)

Architectures supported:
- DRAM-style (periodic word lines, bit lines, contact vias, macro sense-amps, and power mesh)
- FinFET-style (dense vertical fins, transverse gates, standard cell boundaries, and logic cuts)

Physics-based SEM Degradations:
- Secondary Electron (SE) edge blooming (Goldstein et al., 2018)
- Modulation Transfer Function (MTF) point-spread blur (Reimer, 1998)
- Independent Poisson electron beam shot noise + Gaussian transimpedance amplifier readout noise
- Continuous sub-pixel ground-truth coordinate tracking
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def render_dram_layout(x_coords, y_coords, pure_periodic=False):
    """
    Renders realistic DRAM die layout over spatial coordinates (in nanometers).
    """
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # 1. Memory Unit Array (40nm Bitline Pitch x 60nm Wordline Pitch)
    pitch_x, pitch_y = 40.0, 60.0
    dist_x = np.abs((X % pitch_x) - (pitch_x / 2.0))
    bit_lines = (dist_x <= 6.0).astype(np.float32)
    
    dist_y = np.abs((Y % pitch_y) - (pitch_y / 2.0))
    word_lines = (dist_y <= 8.0).astype(np.float32)
    
    via_dist_sq = dist_x**2 + dist_y**2
    vias = (via_dist_sq <= 8.0**2).astype(np.float32)
    
    if pure_periodic:
        base = 0.15 + 0.35 * bit_lines + 0.35 * word_lines + 0.15 * vias
        return np.clip(base, 0.0, 1.0)
        
    # 2. Hierarchical Die Macro-Structures (Sense Amps, Wordline Drivers, Power Grid)
    sense_amp_gap_bool = ((Y % 2500.0) < 220.0)
    wl_driver_gap_bool = ((X % 3200.0) < 260.0)
    
    active_array = (~sense_amp_gap_bool & ~wl_driver_gap_bool).astype(np.float32)
    
    power_rails = ((((Y % 2500.0) >= 60.0) & ((Y % 2500.0) <= 160.0)) |
                   (((X % 3200.0) >= 80.0) & ((X % 3200.0) <= 180.0))).astype(np.float32)
                   
    global_trunk_y = ((np.abs(Y - 5000.0) < 140.0)).astype(np.float32)
    global_trunk_x = ((np.abs(X - 5000.0) < 140.0)).astype(np.float32)
    
    base = (0.15 +
            0.35 * (bit_lines * active_array) +
            0.35 * (word_lines * active_array) +
            0.15 * (vias * active_array) +
            0.30 * power_rails +
            0.40 * (global_trunk_x + global_trunk_y))
            
    return np.clip(base, 0.0, 1.0)


def render_finfet_layout(x_coords, y_coords, pure_periodic=False):
    """
    Renders realistic FinFET standard-cell logic layout over spatial coordinates (in nanometers).
    """
    X, Y = np.meshgrid(x_coords, y_coords)
    
    fin_pitch, fin_width = 28.0, 8.0
    gate_pitch, gate_width = 100.0, 22.0
    
    dist_fin = np.abs((X % fin_pitch) - (fin_pitch / 2.0))
    fins = (dist_fin <= (fin_width / 2.0)).astype(np.float32)
    
    dist_gate = np.abs((Y % gate_pitch) - (gate_pitch / 2.0))
    gates = (dist_gate <= (gate_width / 2.0)).astype(np.float32)
    
    if pure_periodic:
        base = 0.15 + 0.40 * fins + 0.40 * gates
        return np.clip(base, 0.0, 1.0)
        
    cell_boundary_y_bool = ((Y % 1200.0) < 90.0)
    cell_boundary_x_bool = ((X % 1680.0) < 84.0)
    
    gate_cut_y = np.abs((Y % 1200.0) - 600.0) < (gate_width * 0.95)
    gate_cut_x = ((X % 560.0) < 84.0)
    gate_cuts = (gate_cut_y & gate_cut_x).astype(np.float32)
    
    active_gates = np.clip(gates * (~cell_boundary_y_bool).astype(np.float32) - gate_cuts, 0.0, 1.0)
    active_fins = np.clip(fins * (~cell_boundary_x_bool).astype(np.float32), 0.0, 1.0)
    
    power_rails = cell_boundary_y_bool.astype(np.float32) * 0.50
    macro_boundary = (((X < 800.0) | (X > 9200.0) | (Y < 800.0) | (Y > 9200.0))).astype(np.float32) * 0.35
    
    base = 0.12 + 0.35 * active_fins + 0.38 * active_gates + power_rails + macro_boundary
    return np.clip(base, 0.0, 1.0)


def apply_sem_physics(image, is_search=False, edge_weight=0.25,
                      shot_noise_scale=0.03, thermal_sigma=0.02, blur_sigma=0.5):
    """
    Applies rigorous SEM image formation degradation:
    1. Secondary Electron (SE) edge blooming (Goldstein et al., 2018)
    2. Modulation Transfer Function (MTF) beam spot blur (Reimer, 1998)
    3. Poisson beam electron counting shot noise
    4. Gaussian transimpedance amplifier readout noise
    """
    img = image.copy().astype(np.float32)
    
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    grad_mag = np.hypot(gx, gy)
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()
    img = img + edge_weight * grad_mag
    img = np.clip(img, 0.0, 1.0)
    
    if blur_sigma > 0:
        img = ndimage.gaussian_filter(img, sigma=blur_sigma)
        
    if is_search:
        shot_scale = shot_noise_scale * 2.2
        therm_sig = thermal_sigma * 1.8
    else:
        shot_scale = shot_noise_scale
        therm_sig = thermal_sigma
        
    nominal_electrons = 1.0 / (shot_scale**2 + 1e-6)
    poisson_noisy = np.random.poisson(np.maximum(img, 0.01) * nominal_electrons) / nominal_electrons
    gaussian_noise = np.random.normal(0.0, therm_sig, img.shape)
    final_img = poisson_noisy + gaussian_noise
    
    return np.clip(final_img, 0.0, 1.0)


def generate_image_pair(style="dram", center_search_nm=None, drift_sigma_nm=800.0,
                        noise_multiplier=1.0, pure_periodic=False, seed=None):
    """
    Generates a single paired sample (Reference and Search) with continuous ground truth.
    """
    if seed is not None:
        np.random.seed(seed)
        
    search_fov_nm = 10000.0  # 10 µm
    ref_fov_nm = 1000.0      # 1 µm
    search_px = 1000
    ref_px = 1000
    
    margin_nm = ref_fov_nm / 2.0 + 300.0
    if center_search_nm is None:
        cx_nm = 5000.0 + np.random.normal(0.0, drift_sigma_nm)
        cy_nm = 5000.0 + np.random.normal(0.0, drift_sigma_nm)
        cx_nm = np.clip(cx_nm, margin_nm, search_fov_nm - margin_nm)
        cy_nm = np.clip(cy_nm, margin_nm, search_fov_nm - margin_nm)
    else:
        cx_nm, cy_nm = center_search_nm
        
    canvas_px = 2000
    canvas_x = np.linspace(0.0, search_fov_nm, canvas_px, endpoint=False)
    canvas_y = np.linspace(0.0, search_fov_nm, canvas_px, endpoint=False)
    
    ref_x = np.linspace(cx_nm - ref_fov_nm / 2.0, cx_nm + ref_fov_nm / 2.0, ref_px, endpoint=False)
    ref_y = np.linspace(cy_nm - ref_fov_nm / 2.0, cy_nm + ref_fov_nm / 2.0, ref_px, endpoint=False)
    
    if style.lower() == "dram":
        canvas_base = render_dram_layout(canvas_x, canvas_y, pure_periodic=pure_periodic)
        ref_base = render_dram_layout(ref_x, ref_y, pure_periodic=pure_periodic)
    elif style.lower() == "finfet":
        canvas_base = render_finfet_layout(canvas_x, canvas_y, pure_periodic=pure_periodic)
        ref_base = render_finfet_layout(ref_x, ref_y, pure_periodic=pure_periodic)
    else:
        raise ValueError(f"Unknown architecture style: {style}")
        
    if HAS_CV2:
        search_base = cv2.resize(canvas_base, (search_px, search_px), interpolation=cv2.INTER_AREA)
    else:
        search_pil = Image.fromarray((canvas_base * 255.0).astype(np.uint8))
        search_base = np.array(search_pil.resize((search_px, search_px), Image.Resampling.BOX)) / 255.0
    
    ref_sem = apply_sem_physics(
        ref_base, is_search=False,
        edge_weight=0.25,
        shot_noise_scale=0.03 * noise_multiplier,
        thermal_sigma=0.02 * noise_multiplier,
        blur_sigma=0.5
    )
    
    search_sem = apply_sem_physics(
        search_base, is_search=True,
        edge_weight=0.20,
        shot_noise_scale=0.05 * noise_multiplier,
        thermal_sigma=0.04 * noise_multiplier,
        blur_sigma=1.0
    )
    
    gt_x_px = (cx_nm / search_fov_nm) * search_px
    gt_y_px = (cy_nm / search_fov_nm) * search_px
    
    ref_uint8 = (ref_sem * 255.0).round().astype(np.uint8)
    search_uint8 = (search_sem * 255.0).round().astype(np.uint8)
    
    metadata = {
        "style": style,
        "true_center_x": float(gt_x_px),
        "true_center_y": float(gt_y_px),
        "center_nm_x": float(cx_nm),
        "center_nm_y": float(cy_nm),
        "search_fov_nm": search_fov_nm,
        "ref_fov_nm": ref_fov_nm,
        "pure_periodic": pure_periodic,
        "noise_multiplier": noise_multiplier,
        "scale_ratio": 10.0
    }
    
    return ref_uint8, search_uint8, metadata


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Synthetic SEM Dataset Generator")
    parser.add_argument("--style", type=str, default="both", choices=["dram", "finfet", "both"],
                        help="Architecture style: dram, finfet, or both")
    parser.add_argument("--num_pairs", type=int, default=30,
                        help="Number of image pairs to generate (minimum 30 for evaluation)")
    parser.add_argument("--output_dir", type=str, default="./dataset",
                        help="Directory to save generated image pairs")
    parser.add_argument("--noise_multiplier", type=float, default=1.0,
                        help="Noise level scaling factor (1.0 = standard, 1.5+ = challenging test mode)")
    parser.add_argument("--drift_sigma_nm", type=float, default=800.0,
                        help="Standard deviation of stage navigation drift (in nanometers)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    np.random.seed(args.seed)
    manifest = []
    
    styles = ["dram", "finfet"] if args.style == "both" else [args.style]
    
    print(f"[*] Generating {args.num_pairs} image pairs ({args.style}) into '{args.output_dir}'...", flush=True)
    
    for i in range(args.num_pairs):
        sample_style = styles[i % len(styles)]
        sample_dir = os.path.join(args.output_dir, f"pair_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        is_periodic_challenge = (i % 10 == 9)
        
        ref_img, search_img, meta = generate_image_pair(
            style=sample_style,
            drift_sigma_nm=args.drift_sigma_nm,
            noise_multiplier=args.noise_multiplier,
            pure_periodic=is_periodic_challenge
        )
        
        ref_path = os.path.join(sample_dir, "reference.png")
        search_path = os.path.join(sample_dir, "search.png")
        meta_path = os.path.join(sample_dir, "ground_truth.json")
        
        Image.fromarray(ref_img).save(ref_path)
        Image.fromarray(search_img).save(search_path)
        
        meta["pair_id"] = i
        meta["reference_path"] = os.path.abspath(ref_path)
        meta["search_path"] = os.path.abspath(search_path)
        
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        manifest.append(meta)
        
        if (i + 1) % 10 == 0 or (i + 1) == args.num_pairs:
            print(f"    -> Generated {i+1}/{args.num_pairs} pairs [Last GT Center: ({meta['true_center_x']:.2f}, {meta['true_center_y']:.2f})]", flush=True)
            
    summary_path = os.path.join(args.output_dir, "manifest.json")
    with open(summary_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"[+] Dataset generation complete. Manifest saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
