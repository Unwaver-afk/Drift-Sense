# Changes Log

## [v2.0.0] - 2026-08-16

### Added
- **Modular Dataset Generator Package**: Completely overhauled `dataset_generator.py` into a highly extensible 11-module package (`generator/`) satisfying all 25 problem statement deliverables.
  - Added multi-model independent noise module (Poisson + Gaussian + dead-pixel).
  - Added SEM physics modules (edge brightening, anisotropic MTF blur).
  - Added full augmentation stack (rotation, scale jitter, contrast/brightness variance).
  - Added structural randomisation across all dimensions (pitches, widths, gaps).
  - Added manufacturing imperfections (width jitter, pitch jitter, missing/weak features).
  - Added 4 target placement strategies (uniform, edge, center, hard-periodic).
  - Added 4 difficulty tiers determining the probability of augmentation severity.
  - Added rich `manifest.json` ground truth metadata exporting 30+ physical fields including the actual template bounding box.
  - Added automated Quality Control suite verifying boundaries, intensities, and mathematical bounds prior to saving.
  - Added statistical reporting for generated datasets.
- **Docker Support**: Added `Dockerfile` and `docker-compose.yml` for isolated cross-platform execution. Eliminates X11 and MSYS2 environment configuration issues completely.

### Fixed
- **OpenCV Shadowing Issue**: Renamed `cv2.py` to `cv2_shim.py` to prevent it from hijacking Python's import paths. Scripts now reliably use the highly optimized native OpenCV bindings if installed.
- **Dataset Portability**: Fixed `evaluate_benchmarks.py` crashing due to legacy datasets utilizing absolute Windows paths (`E:\Drift Sense\...`). Datasets now use OS-agnostic relative paths.

### Changed
- Re-generated the `benchmark_dataset` entirely, providing 1,000 highly realistic test pairs featuring actual rotational/scale variance and proper splits (`train/`, `validation/`, `hard_cases/`).
- Sub-pixel test baseline performance correctly collapsed to ~4.3% due to the introduction of rotation/scale affine transformations. The `matchTemplate` baseline is now correctly reflecting its physical limits against robust manufacturing defects.
