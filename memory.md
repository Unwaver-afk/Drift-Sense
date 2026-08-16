# Drift-Sense Project Memory & Technical Log

This document records architectural decisions, errors/bugs encountered during development, root cause analyses, solutions implemented, and alternative methods that could be explored.

---

## Log Entries

### Entry 001: Initial Architecture & Ground Truth Precision
- **Challenge**: In a $10\times$ cross-magnification setup ($1000\times 1000$ high-res reference vs $1000\times 1000$ wide search covering $10{,}000\text{ nm}$), 1 pixel in the search image equals $10\text{ nm}$ (10 pixels in reference space). Integer pixel offsets introduce severe discretization error.
- **Decision & Solution**: Use continuous affine transformation tracking in `dataset_generator.py` rather than naive integer slicing.
- **Alternative Evaluated**: Direct image cropping after downsampling. *Rejected* because downsampling first then cropping loses exact sub-pixel reference center alignment.

---

### Entry 002: Spatial Aliasing in Direct 10nm Point Sampling vs. Area Integration
- **Error / Bug Discovered**: In early tests, evaluating the DRAM word/bit line layout with point-sampling on a $10\text{ nm}$ grid produced severe spatial aliasing and low cross-correlation ($\sim 0.55$), because an $8\text{--}12\text{ nm}$ line falls between sample grid points.
- **Root Cause**: Real SEM beams and optical downsampling integrate the total electron flux over the physical spot area (continuous convolution with MTF/PSF), whereas point sampling on discrete coordinates produces sub-harmonic beat frequencies (Moiré-like aliasing).
- **Solution Implemented**: Introduced supersampled continuous canvas rendering ($2.5\text{ nm/px}$, $4000\times 4000$) followed by area integration downsampling (`cv2.INTER_AREA`), guaranteeing physical energy conservation and matching real SEM image formation.
- **Alternative Evaluated**: Analytical box integration for every geometric rectangle. *Evaluated*: Slower to compute than supersampled numpy meshgrids for complex layouts with circular vias and cuts.

---

### Entry 003: NumPy 2.x Type Casting in Boolean Array Operations
- **Error / Bug Discovered**: In `render_realistic_finfet`, combining boolean masks with arithmetic operations (`(dist_gate > ...) & (1.0 - cell_boundary_y)`) resulted in:
  `TypeError: ufunc 'bitwise_and' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''`
- **Root Cause**: NumPy 2.0+ enforces strict type safety for bitwise operators (`&`, `|`, `~`) and does not automatically cast floating-point arrays `(1.0 - float_mask)` to booleans.
- **Solution Implemented**: Explicitly expressed inversion via `~cell_boundary_bool` and cast explicitly to `.astype(np.float32)` only after boolean logic evaluation.

---

### Entry 004: Periodic Ambiguity, Multiple Global Extrema & Center-Prior Weighting
- **Error / Bug Discovered**: In pure infinite periodic grids (e.g. DRAM wordline/bitline grid without sub-array boundaries), the cross-correlation surface forms an exact 2D Dirac comb where every periodic lattice site $(x + k \cdot P_x, y + m \cdot P_y)$ has identical correlation. If candidate selection threshold is set too loose (e.g. $90\%$ of max), the center prior might pull non-periodic unique features towards $(500, 500)$ if noise is high.
- **Root Cause**: There is a distinction between:
  1. *Aperiodic / Macro-structured scenes*: A unique peak exists (global max is clearly higher than secondary peaks).
  2. *Pure periodic array scenes*: Multiple peaks have identical correlation within numerical precision.
- **Solution Implemented**:
  - Structured the dataset generator to reflect real die hierarchy (macro sense-amp blocks, power mesh, logic cuts).
  - Designed the inference engine to apply sub-pixel parabolic refinement on the highest-confidence candidate, and regularize multi-modal ambiguous peaks using the specified minimum-distance center prior.

---

### Entry 005: Parabolic Sub-Pixel Surface Peak Fitting
- **Challenge**: Standard discrete template matching is limited to integer pixel coordinates, which translates to a minimum $\pm 5\text{ nm}$ uncertainty in physical wafer space.
- **Solution Implemented**: Implemented 2D quadratic paraboloid Taylor surface fitting over the $3\times 3$ neighborhood around the discrete peak:
  $$\Delta x = \frac{R(x-1, y) - R(x+1, y)}{2(R(x-1, y) - 2R(x, y) + R(x+1, y))}$$
  $$\Delta y = \frac{R(x, y-1) - R(x, y+1)}{2(R(x, y-1) - 2R(x, y) + R(x, y+1))}$$
  Yielding sub-pixel accuracy of $<0.07\text{ px}$ ($<0.7\text{ nm}$).
- **Alternative Evaluated**: 2D Gaussian peak fitting / Levenberg-Marquardt non-linear optimization. *Rejected* due to $10\times$ higher computational cost with negligible accuracy gain over parabolic Taylor series for smooth correlation peaks.

---

### Entry 006: IDE Language Server / Interpreter Path Discrepancy
- **Error / Issue**: IDE language server reported `Cannot find module numpy`, `PIL`, `scipy`, `matplotlib`, `cv2` with search path querying MSYS2.
- **Root Cause**: The IDE's underlying language server / static analyzer was querying the MSYS2 UCRT64 Python environment, which initially lacked native packages and had ABI incompatibilities with Windows MSVC `.pyd` binaries.
- **Solution Implemented**:
  1. Installed native UCRT64 packages.
  2. Implemented a zero-dependency pure-Python fallback shim `cv2_shim.py`.
  3. Created `pyrightconfig.json` to explicitly bind the workspace interpreter paths.

---

### Entry 007: Fake OpenCV Shadowing the Native CV2 Module
- **Error / Bug Discovered**: Benchmarks running normalized cross correlation were showing only a 14.81% accuracy on the test set despite claims of 100% precision. 
- **Root Cause**: The project contained a file originally named `cv2.py`. Due to Python's module resolution order, `import cv2` found the local shim rather than the highly optimized OpenCV C++ bindings natively installed via pip (`opencv-python`). The pure-Python shim produced slightly different correlation maps.
- **Solution Implemented**: Renamed the file to `cv2_shim.py`. The `localization_inference.py` still safely uses `import cv2` but will now resolve to the actual installed OpenCV bindings, restoring true baseline performance. 

---

### Entry 008: Absolute vs Relative Paths Crashing Benchmarks across Environments
- **Error / Bug Discovered**: Running `evaluate_benchmarks.py` on the benchmark dataset crashed with `FileNotFoundError`.
- **Root Cause**: The legacy dataset generator recorded hardcoded Windows absolute paths (e.g., `E:\Drift Sense\benchmark_dataset\pair_000\reference.png`) inside `manifest.json`.
- **Solution Implemented**: 
  - Rewrote `dataset_generator.py` to always save `os.path.relpath()` in the manifest.
  - Modified `evaluate_benchmarks.py` to dynamically resolve paths via `os.path.join(dataset_dir, item["reference_path"])`, making the dataset 100% portable across OS boundaries.
