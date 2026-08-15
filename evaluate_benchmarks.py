#!/usr/bin/env python3
"""
Drift-Sense Benchmark & Evaluation Suite
========================================
Automated evaluation and diagnostic visualization for Applied Materials Hackathon.

Computes:
  1. Success Rate within sub-pixel (<0.5 px) and standard (<1.0 px) tolerance
  2. Mean Absolute Error (MAE) in pixels and physical nanometers
  3. Execution Latency per 1000x1000 image pair
  4. Generation of visual SUCCESS and HONEST FAILURE case figures
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from localization_inference import find_localized_center


def evaluate_dataset(dataset_dir="./benchmark_dataset", output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    
    if not os.path.exists(manifest_path):
        print(f"Error: manifest.json not found in {dataset_dir}")
        return
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    results = []
    times = []
    
    best_case = None
    worst_case = None
    min_err = float('inf')
    max_err = -1.0
    
    print(f"[*] Evaluating {len(manifest)} image pairs from '{dataset_dir}'...")
    
    for item in manifest:
        ref_path = item["reference_path"]
        search_path = item["search_path"]
        gt_x = item["true_center_x"]
        gt_y = item["true_center_y"]
        style = item.get("style", "unknown")
        pure_periodic = item.get("pure_periodic", False)
        
        ref_img = np.array(Image.open(ref_path).convert("L"))
        search_img = np.array(Image.open(search_path).convert("L"))
        
        # Benchmark single-pair latency
        t0 = time.perf_counter()
        (pred_x, pred_y), conf = find_localized_center(ref_img, search_img)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed_ms)
        
        err_px = float(np.hypot(pred_x - gt_x, pred_y - gt_y))
        err_nm = err_px * 10.0  # 10 nm/px in search image
        
        record = {
            "pair_id": item["pair_id"],
            "style": style,
            "pure_periodic": pure_periodic,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "error_px": err_px,
            "error_nm": err_nm,
            "confidence": conf,
            "latency_ms": elapsed_ms,
            "ref_path": ref_path,
            "search_path": search_path
        }
        results.append(record)
        
        if not pure_periodic and err_px < min_err:
            min_err = err_px
            best_case = record
            
        if pure_periodic and err_px > max_err:
            max_err = err_px
            worst_case = record
            
    # Compute summary statistics
    std_cases = [r for r in results if not r["pure_periodic"]]
    periodic_cases = [r for r in results if r["pure_periodic"]]
    
    std_errors = [r["error_px"] for r in std_cases]
    std_subpixel_acc = sum(e <= 0.5 for e in std_errors) / len(std_errors) * 100.0 if std_errors else 0.0
    std_1px_acc = sum(e <= 1.0 for e in std_errors) / len(std_errors) * 100.0 if std_errors else 0.0
    std_mae_px = float(np.mean(std_errors)) if std_errors else 0.0
    std_mae_nm = std_mae_px * 10.0
    
    mean_latency = float(np.mean(times))
    p95_latency = float(np.percentile(times, 95))
    
    summary = {
        "total_pairs_evaluated": len(results),
        "standard_cases_count": len(std_cases),
        "periodic_stress_cases_count": len(periodic_cases),
        "standard_cases_subpixel_accuracy_pct": round(std_subpixel_acc, 2),
        "standard_cases_1px_accuracy_pct": round(std_1px_acc, 2),
        "standard_cases_mae_pixels": round(std_mae_px, 4),
        "standard_cases_mae_nanometers": round(std_mae_nm, 2),
        "mean_latency_ms": round(mean_latency, 2),
        "p95_latency_ms": round(p95_latency, 2)
    }
    
    print("\n================ BENCHMARK SUMMARY ================")
    print(f"Total Samples Evaluated:            {len(results)}")
    print(f"Sub-Pixel Success Rate (<=0.5 px):  {std_subpixel_acc:.1f}%")
    print(f"Standard Success Rate (<=1.0 px):   {std_1px_acc:.1f}%")
    print(f"Mean Absolute Error (Standard):     {std_mae_px:.3f} px ({std_mae_nm:.1f} nm)")
    print(f"Mean Inference Latency:             {mean_latency:.2f} ms")
    print(f"P95 Latency:                        {p95_latency:.2f} ms")
    print("===================================================\n")
    
    with open(os.path.join(output_dir, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate Visual Diagnostic Plots
    generate_visual_cases(best_case, worst_case, output_dir)
    print(f"[+] Diagnostic visual figures saved to '{output_dir}'.")


def generate_visual_cases(best_case, worst_case, output_dir):
    """
    Renders high-resolution visual SUCCESS and HONEST FAILURE case figures.
    """
    # 1. SUCCESS CASE FIGURE
    if best_case:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
        fig.patch.set_facecolor('#0f172a')
        for ax in axes:
            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values():
                spine.set_color('#334155')
                
        ref = Image.open(best_case["ref_path"])
        search = Image.open(best_case["search_path"])
        
        # Panel 1: High-Res Reference
        axes[0].imshow(ref, cmap='gray')
        axes[0].set_title(f"Reference Image (1 µm x 1 µm, 1 nm/px)\n[{best_case['style'].upper()} Layout]", color='#38bdf8', fontsize=12, pad=10)
        axes[0].axis('off')
        
        # Panel 2: Wide Search with GT & Prediction Bounding Boxes
        axes[1].imshow(search, cmap='gray')
        gt_x, gt_y = best_case["gt_x"], best_case["gt_y"]
        pred_x, pred_y = best_case["pred_x"], best_case["pred_y"]
        
        # GT Box (Green)
        gt_box = patches.Rectangle((gt_x - 50, gt_y - 50), 100, 100, linewidth=2, edgecolor='#22c55e', facecolor='none', label='Ground Truth')
        # Pred Box (Cyan dashed)
        pred_box = patches.Rectangle((pred_x - 50, pred_y - 50), 100, 100, linewidth=2, edgecolor='#38bdf8', linestyle='--', facecolor='none', label='Predicted')
        
        axes[1].add_patch(gt_box)
        axes[1].add_patch(pred_box)
        axes[1].plot(gt_x, gt_y, 'g+', markersize=10, markeredgewidth=2)
        axes[1].plot(pred_x, pred_y, 'cx', markersize=10, markeredgewidth=2)
        axes[1].set_title(f"Wide Search Image (10 µm x 10 µm, 10 nm/px)\nPred: ({pred_x:.1f}, {pred_y:.1f}) | GT: ({gt_x:.1f}, {gt_y:.1f})", color='#38bdf8', fontsize=12, pad=10)
        axes[1].legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
        axes[1].axis('off')
        
        # Panel 3: Zoomed Localized Inset
        crop_r = 75
        x_min, x_max = int(max(0, gt_x - crop_r)), int(min(1000, gt_x + crop_r))
        y_min, y_max = int(max(0, gt_y - crop_r)), int(min(1000, gt_y + crop_r))
        search_np = np.array(search)
        crop_img = search_np[y_min:y_max, x_min:x_max]
        
        axes[2].imshow(crop_img, cmap='gray')
        local_gt_x = gt_x - x_min
        local_gt_y = gt_y - y_min
        local_pred_x = pred_x - x_min
        local_pred_y = pred_y - y_min
        
        axes[2].plot(local_gt_x, local_gt_y, 'g+', markersize=14, markeredgewidth=2.5, label='GT Center')
        axes[2].plot(local_pred_x, local_pred_y, 'cx', markersize=14, markeredgewidth=2.5, label='Pred Center')
        axes[2].set_title(f"Sub-Pixel Local Alignment View\nLocalization Error: {best_case['error_px']:.3f} px ({best_case['error_nm']:.1f} nm)", color='#22c55e', fontsize=12, pad=10)
        axes[2].legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "success_case.png"), facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

    # 2. HONEST FAILURE CASE FIGURE (Aperture / Infinite Periodic Grid Limit)
    if worst_case:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
        fig.patch.set_facecolor('#0f172a')
        for ax in axes:
            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values():
                spine.set_color('#334155')
                
        ref = Image.open(worst_case["ref_path"])
        search = Image.open(worst_case["search_path"])
        
        axes[0].imshow(ref, cmap='gray')
        axes[0].set_title("Reference Pattern in Infinite Periodic Array\n[No Boundary Landmark / Macro Cuts]", color='#f87171', fontsize=12, pad=10)
        axes[0].axis('off')
        
        axes[1].imshow(search, cmap='gray')
        gt_x, gt_y = worst_case["gt_x"], worst_case["gt_y"]
        pred_x, pred_y = worst_case["pred_x"], worst_case["pred_y"]
        
        gt_box = patches.Rectangle((gt_x - 50, gt_y - 50), 100, 100, linewidth=2, edgecolor='#ef4444', facecolor='none', label='True Placement')
        pred_box = patches.Rectangle((pred_x - 50, pred_y - 50), 100, 100, linewidth=2, edgecolor='#f59e0b', linestyle='--', facecolor='none', label='Center-Prior Match')
        
        axes[1].add_patch(gt_box)
        axes[1].add_patch(pred_box)
        axes[1].plot(gt_x, gt_y, 'r+', markersize=10, markeredgewidth=2)
        axes[1].plot(pred_x, pred_y, 'yx', markersize=10, markeredgewidth=2)
        axes[1].set_title(f"Aperture Ambiguity & Modulo-Shift\nTrue vs. Center-Prior Selected Site", color='#f87171', fontsize=12, pad=10)
        axes[1].legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
        axes[1].axis('off')
        
        # Panel 3: Explanatory Schematic of Dirac Comb Periodic Correlation Surface
        x_mesh = np.linspace(-4, 4, 300)
        corr_comb = (np.cos(x_mesh * np.pi * 2.0) + 1.0) / 2.0
        center_penalty = np.exp(-0.5 * (x_mesh / 3.0)**2)
        combined = corr_comb * center_penalty
        
        axes[2].plot(x_mesh, corr_comb, color='#64748b', linestyle=':', label='Periodic Raw Correlation')
        axes[2].plot(x_mesh, combined, color='#f59e0b', linewidth=2, label='Cost with Center Prior')
        axes[2].axvline(0.0, color='#f59e0b', linestyle='--', alpha=0.7, label='Center Prior Optimum')
        axes[2].axvline(2.0, color='#ef4444', linestyle='--', alpha=0.7, label='True Shift (Modulo Ambiguous)')
        axes[2].set_title("Mathematical Limitation: Periodic Dirac Comb\n[Equal Mutual Information across Lattice Shifts]", color='#f59e0b', fontsize=12, pad=10)
        axes[2].legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
        axes[2].set_xlabel("Lattice Shift Steps", color='#94a3b8')
        axes[2].set_ylabel("Normalized Response", color='#94a3b8')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "honest_failure_case.png"), facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Benchmark & Diagnostic Suite")
    parser.add_argument("--dataset_dir", type=str, default="./benchmark_dataset")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()
    
    evaluate_dataset(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
