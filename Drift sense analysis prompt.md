# Prompt: Analyze the Drift-Sense Problem Statement

You are a senior software/algorithms engineer at a semiconductor equipment company, reviewing a hackathon problem statement before your team commits engineering time to it. Read the full problem statement below and produce a rigorous technical analysis — not a restatement of it, not a pitch, not code yet. Your job is to find where the real difficulty is, where teams will get it wrong, and what a winning approach actually requires.

## Problem statement

**Title:** Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

**Background:**
A modern wafer inspection tool must return to exactly the same site on a die — thousands of times per day, across hundreds of dies on every wafer. The first visit is straightforward: the tool finds the target site and records its location. Every subsequent visit must land at the identical spot with nanometre accuracy so that measurements are comparable across time and across different tools.

In practice, this does not happen perfectly. Motion stages accumulate tiny errors between visits — drift from thermal expansion, vibration from the fab environment, and mechanical slack. These errors compound over time. By the time the tool revisits a site on a different die (or on the same die hours later), it may land several pixels away from the intended location.

Because all dies on a wafer carry the same repeating circuit layout, the tool cannot simply look at the landed image and know it is wrong — the structure at the wrong location looks almost identical to the correct one. This is the core difficulty: finding a specific site inside a sea of visually similar repeating patterns.

Applied Materials calls this challenge Navigation-Error Recovery. It is solved today with classical template-matching algorithms, but these break down in highly periodic layouts (DRAM memory arrays, FinFET gate structures) where hundreds of nearly identical features are visible in the same frame. AI and advanced computer vision approaches offer a path to more robust, higher-accuracy recovery.

**What your algorithm must produce:**
- Find the location inside the Search Image where the Reference Image pattern appears (shrunk 10x).
- Return the center coordinates (x, y) in pixels of the matching region within the Search Image.
- If more than one matching region is found, return the one closest to the center of the Search Image.

**Scale/geometry:**
- Both the reference and wide-search images are grayscale, 1000×1000 pixels.
- At the high-resolution ("100x") capture, pixel size is 1 nm — a 1 µm × 1 µm field of view.
- The wide-search ("10x") capture covers exactly 10x the physical area at the same pixel count (10 nm/pixel, ~10 µm × 10 µm field of view) — the reference pattern appears shrunk by that same 10x factor somewhere inside it.
- This is a search-and-localize problem: finding a known pattern, at a known but reduced scale, inside a noisier, lower-resolution image that may also contain visually similar repeating structures — since DRAM- and FinFET-style layouts are themselves highly periodic, which is what makes this genuinely hard rather than a simple exact-pixel lookup.

**Dataset — no dataset is provided; you must generate synthetic data:**
Each training sample is an image pair: a Reference Image and a Search Image. Choose either DRAM-style or FinFET-style die architecture — both judged equally.

- *DRAM-style:* Reference = periodic horizontal word-lines and vertical bit-lines crossing at right angles, with a small contact/via dot at every intersection; high-contrast, fine pitch, extremely regular. Search = a larger tiled version of the same DRAM grid, downsampled to 1000×1000 to represent the 10x lower magnification; the reference pattern appears shrunk ~10x somewhere inside.
- *FinFET-style:* Reference = dense parallel vertical fin lines, crossed by one or two horizontal gate bars at the intersection region; high-contrast vertical structure with distinctive gate crossings. Search = same approach — a larger tiled FinFET layout, downsampled to 1000×1000; the reference appears as a small ~100×100 pixel inset somewhere inside.

**Mandatory requirements for the dataset generator:**
- Generate independent sensor noise for each image — do NOT reuse the same noise on both; they are separate physical captures with independent noise patterns.
- Apply edge-brightening to mimic real SEM behavior — SEM images show brighter contrast along feature edges.
- The true location of the reference pattern within the search image must be known and recorded — this is the ground truth for computing accuracy.
- Include realistic degradation: blur, rotation, and scaling variations. The search image will have more noise than the reference image on actual test data — design accordingly.
- Generate a minimum of 30 randomized image pairs for self-evaluation. Applied Materials will evaluate the algorithm on a separate set of generated test cases.

**Citation requirement:** every augmentation choice, noise model, and structural parameter selected must be justified with at least 2–3 credible public references — academic papers, textbooks, or patents on semiconductor device structure or SEM imaging. Unjustified augmentation choices will not receive full marks on the 30% augmentation score.

**Starter support:** Applied Materials will provide a starter Python prompt (PIL/numpy) generating one basic image pair showing the correct 1000×1000 structure, 10x zoom relationship, and independent noise. This is a starting scaffold, not a complete dataset generator.

**Test data — what Applied Materials will use for Phase 2:**
- Generated using a similar approach to participants, but with parameters, noise levels, and exact placement coordinates known only to Applied Materials.
- At least 30 randomized image pairs covering both DRAM-style and FinFET-style layouts.
- MORE noisy than the training examples — search images in the test set will have higher noise levels to test algorithm robustness.
- Includes at least one highly periodic array region where correct localization is genuinely difficult — specifically designed to test failure-mode awareness.

**Expected solution:**
- Build a synthetic-but-realistic grayscale dataset generator producing reference and wide-search image pairs mimicking either DRAM-style or FinFET-style die architecture — participant's choice, judged equally either way — using only publicly known structural characteristics, never proprietary fab data.
- Justify every augmentation/noise choice, distortion, rotation & scaling against at least 2–3 credible public sources, cited in the final presentation.
- Correctly account for the true 10x zoom ratio between the high-resolution reference and the wide-search image, then use a classical ML or DL-based localization algorithm of choice to find the reference pattern and report the center (x, y) of the matching region — or, if more than one region matches, whichever is closest to the search image's center.
- Report a measurable success rate: (1) computation time of algorithm on a single image (1k×1k), (2) run the method across at least 30 randomized generated test cases and report the percentage landing within a stated tolerance (e.g., within subpixel of the true downsampled location), plus at least one honest example of where it fails (e.g., inside a highly periodic array region) and why.
- Bonus: solutions that additionally generalize to optical microscope images (RGB, 3-channel) — not just the primary SEM grayscale case — earn bonus credit, provided the core SEM-based solution is completed first.

**Submission requirements — Component 1 (PPT/PDF via i4C Idea Submission Template):**
1. Team Details — team name, member names, roles, college name, contact details.
2. Problem Statement Addressed — describe in your own words why navigation-error recovery matters in semiconductor wafer inspection, using the Background section as context.
3. Idea Description — key concept and approach: DRAM-style or FinFET-style? Which localization algorithm (classical ML or deep learning)? Why is the approach better than simple template matching for periodic layouts?
4. Proposed Solution — dataset generator design (architecture, noise models, augmentation), localization algorithm (architecture/method, key design decisions), pipeline diagram from input pair to output (x, y). Include citations for every augmentation choice.
5. Innovation & Uniqueness — what makes the approach different? Is the dataset generator more realistic than the baseline? Does the localization handle periodic ambiguity better? Any novel approach to the 10x scale-difference problem?
6. Results — accuracy rate on the team's own 30+ test cases (percentage within tolerance of true location), computation time per image pair (1000×1000), one visual SUCCESS case and one visual HONEST FAILURE case (reference image, search image, predicted location, true location).
7. Technology & Feasibility — tech stack, hardware used for development (CPU/GPU, cloud), dataset generation time, localization inference time per pair, model size if a DL method is used.
8. GitHub & Video Link — GitHub repository link (mandatory); video link showing the algorithm running on a sample pair (optional but recommended).
9. References — all citations used to justify augmentation/noise choices; research papers, patents, or textbooks on SEM imaging, semiconductor structure, or computer vision methods used.

**Submission requirements — Component 2 (GitHub repository, mandatory, must be public):**
1. `README.md` — complete setup instructions; a reviewer must be able to clone the repo, generate a sample image pair, and run the localization algorithm from the README without contacting the team.
2. Dataset generator script (standalone `.py`) — documented; must accept parameters (architecture style DRAM/FinFET, number of pairs to generate, output directory); must record the true center coordinates of the reference pattern in each generated pair as ground truth.
3. Localization inference script (standalone `.py`) — accepts (a) path to reference image, (b) path to search image; outputs the predicted center (x, y) of the reference pattern within the search image; must run without manual edits — this is the script Applied Materials will run on test data.
4. DL model weights (if applicable) — downloadable format (`.pt`, `.h5`, `.onnx`); must load automatically in the inference script.
5. Training script or notebook (if applicable) — reproduces the training process.
6. `requirements.txt` — complete `pip freeze` output from the development environment, required for reproducibility.
7. Citation documents / supporting references — PDF or markdown listing all references used to justify augmentation and noise choices; must correspond to the citations in the PPT.

**Critical note:** the localization inference script is the most important file in the repository. Applied Materials will run it directly on their test image pairs to compute the Phase 2 score. It must run without manual edits, must accept a reference image path and search image path as inputs, and must output a single (x, y) coordinate. Test it on a fresh machine before submitting — an unrunnable script cannot be scored.

## What your analysis must cover

1. **Restate the actual technical problem in one or two sentences** — strip away the hackathon framing and state precisely what class of problem this is (e.g. known-scale localization under periodic ambiguity), and which parts of the naive framing (scale search, arbitrary rotation) are red herrings versus genuine constraints.
2. **Rank the real technical difficulties** — periodicity/self-similarity, noise asymmetry between captures, sub-pixel accuracy requirements, multi-match tie-breaking, runtime constraints — and explain concretely why each one breaks a naive solution.
3. **Compare candidate algorithmic approaches** (classical template matching, FFT/phase correlation, feature-based matching, learned embeddings) against the specific failure modes above — not in the abstract, but against *this* dataset's periodicity and noise profile. State which approach you'd actually ship and why.
4. **Flag the dataset-generator pitfalls** that would silently invalidate the ground truth or cost points on the augmentation-justification score (coordinate bookkeeping through transforms, independent noise, citation-worthy augmentation choices, realistic structural parameters).
5. **Flag the grading-critical execution details** — what makes the inference script fail Applied Materials' automated scoring, and what the "honest failure case" requirement is actually testing for (understanding of a fundamental, not just implementation, limitation).
6. **Identify where the real differentiation between submissions will happen** — what most teams will get wrong by default, and what's cheap to get right but expensive to fake credibly in a short writeup.

Do not write implementation code in this pass. Output a structured technical analysis a team can use to decide its architecture and division of labor before writing a single line of the actual solution.