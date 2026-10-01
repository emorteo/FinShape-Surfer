# FinShape Surfer

FinShape Surfer is a scientific Python application for manual dolphin dorsal-fin
morphometrics, reproducible 2D reconstruction, rotatable 3D visualization,
Photo-ID inspection, and experimental CFD surface preparation. The repository
also retains the earlier modular FinShape3D prototype and its tests.

Note: the standalone `finshape3d` is not an entrypoint anymore; use the active
GUI (`finshape_surfer_gui.py`) or call the library functions directly (see
"Programmatic usage" below).

## Highlights
- Modular `core/` package: `math`, `io`, `reconstruction`, `rendering`,
  `vision`, and `ui`.
- FinShape Surfer 1.3.13 GUI derived from the frozen 1.2.2 release:
  `finshape_surfer_gui.py`.
- Programmatic reconstruction API in `core.reconstruction.wireframe`.
- Optional, operator-reviewed FinFindR crop and Photo-ID contour suggestions.

## Requirements
- Python 3.8 or newer
- Recommended: create a virtual environment

Optional (improves GUI functionality):
- `Pillow` — image transforms, raster export

Install dependencies (recommended):
```bash
# from project root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer to install the package in editable mode:
```bash
pip install -e .
```

For a complete GUI environment with experimental FinFindR assistance,
high-resolution photo display, and closed-surface export, install the `full`
profile and prepare the exact external model checkout:

```bash
pip install -e '.[full]'
python tools/prepare_finfindr_models.py
```

The narrower `finfindr` profile installs only learned-model inference support.
The `surface` profile installs the pinned Shapely and trimesh versions required
by high-resolution photo display and experimental surface export.

See [FinFindR-assisted vision](docs/FINFINDR_ASSISTED_VISION.md) for model
provenance, setup, licensing boundaries, validation status, and limitations.

## Current GUI behavior

The active entry point is `finshape_surfer_gui.py` (version 1.3.13). The
interface preserves the manual 1.2.2 measurement workflow and adds explicitly
reviewable assistance:

- **Edit image:** `1. Load Image`, `2. Flip`, `3. Rotate`, `4. Crop`, and
  `5. Apply Crop`. The source filename is shown in the measurement workspace.
- **Morphometry:** `6. Set Landmarks`, `7. Save Measurement (CSV)`, `8. Show
  2D`, `9. Show 3D`, `10. Measure new image`, and `Exit (Save CSV)`.
- **Assisted vision:** `Automatic fin detection`, `Apply detected crop`, and
  `Automatic edge detection`. Detection results are review layers; accepting a
  crop is a separate action.

Landmarks are placed in the order `A -> B -> C30 -> C20 -> C10 -> C5 -> D`.
The measured landmarks, Bezier controls, CSV values, and CFD boundary remain
authoritative. The reviewed B-to-A-to-C30 contour is used by Photo-ID display
masking to remove residual water and preserve identity-bearing nicks and
notches; it does not replace the measured morphometry.

The workflow credits are displayed in the corresponding rows: image
algorithms are attributed to FinShape2D R-GUI 2.0.20, Eduardo Morteo, and
Rodrigo Morteo; morphometric ratios to David Weller and Eduardo Morteo in
Morteo et al. (2017); and assisted vision to FinFindR 0.1.21 and Jaime
Thompson.

Version 1.3.13 changes only interactive Photo-ID rendering and the display tip
closure. While Pitch, Yaw, Roll, or direct drag is active, Photo-ID uses a
120,000-pixel, single-sample preview on a cached boundary-preserving display
mesh; releasing the control immediately restores the existing full mesh,
420,000-pixel, 2x-antialiased view. Saved 1920 x 1440 output is unchanged. The
ruled tip retains its measured planform targets and wire
topology, but now closes thickness on a quarter-elliptic envelope instead of a
linear taper, producing a convex airfoil-like end profile. Measurement, Bezier,
Photo-ID masking, and CFD export geometry remain unchanged.

Version 1.3.12 replaces the insufficient apex-omission rule with a closed,
ruled airfoil-tip grid. Longitudinal rails continue from one common body chord
to distinct, equally spaced positions on the measured tip contour; transverse
rings are evaluations of the same surface and therefore meet every rail
exactly. Incomplete legacy constant-thickness diagonals are removed from the
display. Multi-interval Photo-ID masks now use a dense constrained-Delaunay
triangulation instead of stretching a few coarse image triangles, while robust
Lab-color evidence gives dark-blue water versus dark-gray tissue boundaries
more influence during source-resolution contour refinement. These changes are
display-only; measured Beziers and the existing CFD exporter are unchanged.

Version 1.3.11 removes the remaining apex pole from the displayed wire grid:
all longitudinal rails stop on a common cap section and sparse transverse
rings describe the measured rounded tip without converging blue lines. The
Photo-ID path restores 1.3.8's dense 321-by-41 structured airfoil surface for
ordinary single-chord fin sections, eliminating the coarse faceted silhouette
regression introduced in 1.3.10. A boundary-conforming fallback remains only
for genuine multi-interval sections, so deep missing-tissue contours are not
bridged. Regression tests now inspect rendered apex pixels, 3D line artists,
the selected mesh topology, and an oblique rasterized silhouette. All three
workflow credit labels are right-aligned at the same default font size.

Version 1.3.10 attempted to retain one center apex seam and made the arbitrary
boundary triangulation the default Photo-ID mesh. Reported real-case review
showed that the seam still formed a visible fan and that the coarse mesh
reintroduced a serrated oblique silhouette; 1.3.11 supersedes both choices.

Version 1.3.9 moves the source filename into the measurement workspace and
adds explicit image-algorithm and morphometry attribution to their respective
workflow rows. The wireframe now clips interior longitudinal display rails to
a convex airfoil-tip envelope instead of drawing every rail through the apex;
the measured surface and CFD export are unchanged. Automatic edge detection
also refreshes an already-saved Photo-ID snapshot immediately, so its detailed
trailing boundary removes residual water even when detection is run after Save
Measurement. Interactive Photo-ID frames use a bounded preview buffer while
saved 1920 x 1440 renders retain the full existing quality settings.

Version 1.3.8 restores post-save review navigation: closing the 2D window
returns focus to the measurement window and **Automatic edge detection** stays
available after the measurement is saved. It also fixes Photo-ID reconstruction
when the attachment chord lies exactly on the planform boundary by using the
authoritative B-C30 endpoints for that row instead of a numerically fragile
line/polygon intersection. The validated experimental CFD exporter remains
separate from the display airfoil mesh; see the assisted-vision design notes
for the geometry assessment.

Version 1.3.7 removes the separate tip cap and textured silhouette ribbon from
1.3.6. Both were rejected because they could create concave tip geometry and a
duplicate leading edge while leaving the underlying coarse surface visible.
Wireframe and Photo-ID views now derive thickness from closed symmetric
NACA-style sections that taper with the local measured chord and close to zero
at both contour edges and the apex. Photo-ID uses one structured span/chord
mesh, so the photographed skin and its silhouette are the same geometry. No
auxiliary ribbon, cap, or duplicate surface is rendered. The measured planform
remains authoritative and experimental CFD export remains unchanged pending
separate solver-oriented validation.

Version 1.3.6 attempted an explicit rounded cap grid and a separate textured
leading-edge ribbon. Real-case review rejected both approaches; they are
retained in this history only to document why they were removed in 1.3.7.

Version 1.3.5 retained the 1.3.4 multiscale and color-assisted detection while
repairing five review-display regressions. Embedded instruction figures now
survive text corrections, draggable overlays are replaced cleanly on every
frame, and the freely movable C5-C20 handle drives two smooth quadratics that
remain anchored at fixed C10 without changing the untouched inherited fit.
Longitudinal display rails stopped before the zero-width apex and closely
spaced transverse sections described the rounded tip without a triangular
fan. The Photo-ID mask combined the operator-
fitted smooth B-to-D-to-A leading edge with the detailed FinFindR A-to-C30
trailing edge, intending to eliminate learned pixel stairs while preserving individual
nicks and notches. These display changes do not alter the frozen 1.2.2 source.

Version 1.3.4 added multiscale fallback detection for small fins and combined
FinFindR edge probabilities with source-resolution gradients and robust color
evidence learned from the operator-fitted fin interior and outer crop frame.
The resulting B-to-A-to-C30 contour is an operator-reviewed Photo-ID mask only:
it removes water around nicks and notches without replacing the measured Bezier
wireframe or CFD boundary. Leading-edge display smoothing suppresses raster
staircases while trailing-edge detail is retained. Wireframe rails and height
sections are uniformly limited for readability without changing the simulation
mesh. The 3D convention is Left = +Z and Right = -Z.

## Running the GUI (Quick Start)
1. Activate your virtualenv (see above).
2. Launch the GUI:
```bash
python finshape_surfer_gui.py
```
3. Workflow (GUI):
	- Load an image.
	- Optionally click `Automatic fin detection`, review the green crop, and
	  click `Apply detected crop` only when the proposal is acceptable.
	- Set landmarks in order: A → B → C30 → C20 → C10 → C5 → D.
	- Optionally click `Automatic edge detection` after A, B, and C30 are set.
	- Adjust Bézier controls if needed, then click `7. Save Measurement (CSV)`.
	- Use `8. Show 2D` and `9. Show 3D` to inspect the measured wireframe and
	  Photo-ID view. Closing a derived view returns to the measurement workspace.

Versioned outputs are written next to the source image in
`<image-stem>_FinShape2D3D_outputs_v1.3.13/`. The CFD export remains separate
from the display-only airfoil-style Photo-ID reconstruction.

## Programmatic usage
If you want to run reconstruction from a script or pipeline, use the library API:

```python
from pathlib import Path
import pandas as pd
from core.reconstruction.wireframe import build_landmarks, build_controls, compute_2d_curves

row = pd.Series(...)  # a row with r_BA,r_BC5,r_BC10,r_BC20,r_BC30 or C*B_over_AB fields
lm, u_AB = build_landmarks(row, BC30_scale=10.0)
ctrl = build_controls(lm, u_AB)
curves2d = compute_2d_curves(ctrl)
# plotting helpers are available in core.ui.plotting_3d
```

## Preserved FinShape Surfer baseline

The reviewed FinShape Surfer 1.2.2 manual workflow is preserved separately under `baselines/FinShape_Surfer_1.2.2/`. Its golden numerical regression fixture and tests are documented in `docs/BASELINE_FREEZE_1.2.2.md`. This does not replace the older modular prototype at the repository root.

New work is developed on focused branches in separate Git worktrees. See
[`docs/BRANCH_AND_WORKTREE_WORKFLOW.md`](docs/BRANCH_AND_WORKTREE_WORKFLOW.md).
The first external-algorithm reference harness is documented under
[`references/finfindr-0.1.21/`](references/finfindr-0.1.21/).

## Testing
Run the project's test suite with `pytest`:

```bash
pytest -q
```

Tests use pytest fixtures and create temporary files via `tmp_path`; they do
not modify the repository tree. The full suite includes deterministic tests
for the pinned FinFindR adapter and display topology. Tests that execute the
external checkpoint require the prepared model directory:

```bash
FINSHAPE_FINFINR_MODEL_DIR=/path/to/finfindr-0.1.21/source/inst/extdata pytest -q
```

The GUI never downloads model files at runtime. A release validation run with
the pinned checkpoint must report the complete suite as passing; without the
model directory, only the model-dependent tests should be skipped.

## Development practices
- Use the provided `.gitignore` to avoid committing virtualenvs, build artifacts and generated outputs.
- Place private or local source material under `data/`; its contents are ignored
  while `data/.gitkeep` preserves the directory. Put local generated exports
  under `outputs/`, which follows the same policy.
- Store only reviewed, redistributable public regression inputs under
  `tests/fixtures/`. Do not commit restricted research photographs or catalog
  data.

## Security and research data

Read the [public-release security review](docs/PUBLIC_RELEASE_SECURITY_REVIEW.md)
before publishing this working copy. Generated provenance manifests and surface
packages may contain absolute source paths and catalog identifiers; inspect and
sanitize them before external distribution.

Do not report credentials or restricted research data in a public issue. Rotate
any credential immediately if it is accidentally committed, because deleting it
from the latest revision does not remove it from Git history.

## Troubleshooting
- If `Show 3D` reports a geometry error, check the CSV row printed by the GUI
  error dialog. Noisy ratio inputs can violate geometric constraints; the
  wireframe view remains available so the measurement data is not lost.
- If image open fails on Linux, ensure `xdg-open` is available or open the output manually.

## Disclaimer
This software is research-oriented and provided "as-is" for academic and development use. There is no warranty of accuracy, fitness for a particular purpose, or non-infringement. Use at your own risk. The authors and contributors are not liable for any damages arising from its use.

## Contributing
Contributions and issues are welcome. Please follow standard GitHub workflows: open an issue to discuss major changes, then submit a pull request with tests and documentation updates.

## License

Copyright (c) 2026 Eduardo Morteo. All contents of this repository are
available under the [MIT License](LICENSE).

Optional external FinFindR source and checkpoints are not repository contents
and retain their upstream license. See [Third-party notices](THIRD_PARTY_NOTICES.md).

Contributors must not add photographs, research data, third-party assets, or
other material unless they have authority to distribute it under this license.
