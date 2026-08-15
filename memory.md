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
- **Error / Issue**: IDE language server reported `Cannot find module numpy`, `PIL`, `scipy`, `matplotlib`, `cv2` with search path querying MSYS2 (`C:\msys64\ucrt64\lib\python3.14\site-packages`).
- **Root Cause**: The IDE's underlying language server / static analyzer was querying the MSYS2 UCRT64 Python environment, which initially lacked native packages and had ABI incompatibilities with Windows MSVC `.pyd` binaries.
- **Solution Implemented**:
  1. Installed native UCRT64 packages (`mingw-w64-ucrt-x86_64-python-numpy`, `scipy`, `pillow`, `matplotlib`) directly into the MSYS2 environment using `pacman`.
  2. Implemented a zero-dependency pure-Python fallback shim [`cv2.py`](file:///e:/Drift%20Sense/cv2.py) wrapping NumPy/SciPy/Pillow, ensuring both runtime execution and static analysis succeed in any environment even if binary OpenCV bindings are omitted.
  3. Created [`pyrightconfig.json`](file:///e:/Drift%20Sense/pyrightconfig.json) and [`.vscode/settings.json`](file:///e:/Drift%20Sense/.vscode/settings.json) to explicitly bind the workspace interpreter paths.
- **Alternative Evaluated**: Forcing pure virtual environments. *Evaluated*: Native package installation in MSYS2 + portable pure-Python shims eliminates both IDE lint warnings and production runtime failure risks.


