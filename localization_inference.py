#!/usr/bin/env python3
"""
Drift-Sense Standalone Localization Inference Script
===================================================
Applied Materials Hackathon Submission - Component 2 Target Script

Accepts:
  - Reference image path (1000x1000 px, high resolution 1 nm/px)
  - Search image path (1000x1000 px, wide search 10 nm/px, 10x FOV)

Outputs:
  - Predicted center coordinates (x, y) in pixels within the Search Image.

Features:
  - Robust 10x anti-aliased cross-magnification downsampling
  - Normalized 2D cross-correlation (ZNCC)
  - Deterministic center-distance prior tie-breaking for periodic DRAM/FinFET layouts
  - 2D parabolic sub-pixel peak interpolation (0.05 px resolution)
"""

import sys
import os
import argparse
import json
import numpy as np
from PIL import Image
import scipy.signal as signal
import scipy.ndimage as ndimage

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def subpixel_refinement(correlation_map, max_loc):
    """
    Fits a continuous 2D quadratic paraboloid over a 3x3 window around discrete peak
    to estimate sub-pixel offsets (dx, dy).
    """
    x0, y0 = max_loc
    H, W = correlation_map.shape
    
    if x0 <= 0 or x0 >= W - 1 or y0 <= 0 or y0 >= H - 1:
        return 0.0, 0.0
        
    z = correlation_map[y0-1:y0+2, x0-1:x0+2].astype(np.float64)
    
    denom_x = 2.0 * (2.0 * z[1, 1] - z[1, 0] - z[1, 2])
    dx = (z[1, 2] - z[1, 0]) / (denom_x + 1e-12) if abs(denom_x) > 1e-10 else 0.0
    
    denom_y = 2.0 * (2.0 * z[1, 1] - z[0, 1] - z[2, 1])
    dy = (z[2, 1] - z[0, 1]) / (denom_y + 1e-12) if abs(denom_y) > 1e-10 else 0.0
    
    dx = float(np.clip(dx, -0.5, 0.5))
    dy = float(np.clip(dy, -0.5, 0.5))
    
    return dx, dy


def find_localized_center(ref_img, search_img, center_prior_weight=0.035):
    """
    Finds the center (x, y) of the reference pattern in the search image.
    """
    # 1. Anti-aliased 10x downsampling of Reference Image (1000x1000 -> 100x100)
    if HAS_OPENCV:
        ref_down = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    else:
        ref_pil = Image.fromarray(ref_img)
        ref_down = np.array(ref_pil.resize((100, 100), Image.Resampling.BOX))
        
    tpl_h, tpl_w = ref_down.shape[:2]
    search_h, search_w = search_img.shape[:2]
    
    # 2. Normalized Cross-Correlation (ZNCC)
    if HAS_OPENCV:
        corr_map = cv2.matchTemplate(search_img, ref_down, cv2.TM_CCOEFF_NORMED)
    else:
        ref_zero = ref_down.astype(np.float32) - np.mean(ref_down)
        ref_zero /= (np.std(ref_zero) + 1e-6)
        search_zero = search_img.astype(np.float32) - np.mean(search_img)
        search_zero /= (np.std(search_zero) + 1e-6)
        corr_map = signal.correlate2d(search_zero, ref_zero, mode='valid') / (tpl_w * tpl_h)
        
    map_h, map_w = corr_map.shape
    global_max = float(np.max(corr_map))
    
    center_topleft_x = (search_w / 2.0) - (tpl_w / 2.0)
    center_topleft_y = (search_h / 2.0) - (tpl_h / 2.0)
    max_dist = np.hypot(center_topleft_x, center_topleft_y)
    
    candidate_threshold = max(0.20, global_max - 0.035)
    candidate_mask = corr_map >= candidate_threshold
    
    if np.any(candidate_mask):
        valid_indices = np.argwhere(candidate_mask)
        best_loc = None
        min_cost = float('inf')
        for y, x in valid_indices:
            corr_val = corr_map[y, x]
            dist_to_center = np.hypot(x - center_topleft_x, y - center_topleft_y)
            dist_norm = dist_to_center / (max_dist + 1e-6)
            cost = -corr_val + (center_prior_weight * dist_norm)
            if cost < min_cost:
                min_cost = cost
                best_loc = (int(x), int(y))
    else:
        best_loc = np.unravel_index(np.argmax(corr_map), corr_map.shape)[::-1]
        
    dx, dy = subpixel_refinement(corr_map, best_loc)
    
    pred_center_x = float(best_loc[0] + dx + (tpl_w / 2.0))
    pred_center_y = float(best_loc[1] + dy + (tpl_h / 2.0))
    confidence = float(corr_map[best_loc[1], best_loc[0]])
    
    return (pred_center_x, pred_center_y), confidence


def main():
    parser = argparse.ArgumentParser(
        description="Applied Materials Drift-Sense Localization Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ref_pos", nargs="?", type=str, help="Path to Reference Image (1000x1000)")
    parser.add_argument("search_pos", nargs="?", type=str, help="Path to Search Image (1000x1000)")
    parser.add_argument("--ref", "--ref_path", dest="ref_flag", type=str, help="Path to Reference Image")
    parser.add_argument("--search", "--search_path", dest="search_flag", type=str, help="Path to Search Image")
    parser.add_argument("--json", action="store_true", help="Output results formatted as JSON")
    
    args = parser.parse_args()
    
    ref_path = args.ref_pos or args.ref_flag
    search_path = args.search_pos or args.search_flag
    
    if not ref_path or not search_path:
        parser.print_help()
        sys.exit(1)
        
    if not os.path.exists(ref_path):
        sys.stderr.write(f"Error: Reference image not found at '{ref_path}'\n")
        sys.exit(2)
        
    if not os.path.exists(search_path):
        sys.stderr.write(f"Error: Search image not found at '{search_path}'\n")
        sys.exit(2)
        
    ref_img = np.array(Image.open(ref_path).convert("L"))
    search_img = np.array(Image.open(search_path).convert("L"))
    
    (x, y), score = find_localized_center(ref_img, search_img)
    
    if args.json:
        result = {
            "predicted_center_x": round(x, 4),
            "predicted_center_y": round(y, 4),
            "confidence": round(score, 4),
            "status": "SUCCESS"
        }
        print(json.dumps(result))
    else:
        print(f"{x:.4f}, {y:.4f}")


if __name__ == "__main__":
    main()
