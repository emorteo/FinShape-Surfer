# FinShape Surfer

FinShape Surfer is a scientific Python application for manual dolphin dorsal-fin
morphometrics, reproducible 2D reconstruction, rotatable 3D visualization,
Photo-ID inspection, and experimental CFD surface preparation. The repository
also retains the earlier modular FinShape3D prototype and its tests.

Note: the standalone `finshape3d` is not an entrypoint anymore; use the GUI (`finshape2d.py`) or call the library functions directly (see "Programmatic usage" below).

## Highlights
- Modular `core/` package: `math`, `io`, `reconstruction`, `ui`.
- Interactive 2D landmarking GUI: `finshape2d.py`.
- Programmatic reconstruction API in `core.reconstruction.wireframe`.

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

## Running the GUI (Quick Start)
1. Activate your virtualenv (see above).
2. Launch the GUI:
```bash
python finshape2d.py
```
3. Workflow (GUI):
	- Load an image.
	- Set landmarks in order: B → A → C30 → C20 → C10 → C5 → D.
	- Adjust Bézier controls if needed, then click `Save Measurement (CSV)`.
	- After saving, click `Generate 3D` to produce 2D/3D outputs in `FinShape3D_outputs/` next to the image file.

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

## Testing
Run the project's test suite with `pytest`:

```bash
pytest -q
```

Tests use pytest fixtures and create temporary files via `tmp_path`; they do not modify the repository tree.

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
- If `Generate 3D` fails with geometry errors, check the CSV row printed by the GUI error dialog — noisy ratio inputs can violate geometric constraints.
- If image open fails on Linux, ensure `xdg-open` is available or open the output manually.

## Disclaimer
This software is research-oriented and provided "as-is" for academic and development use. There is no warranty of accuracy, fitness for a particular purpose, or non-infringement. Use at your own risk. The authors and contributors are not liable for any damages arising from its use.

## Contributing
Contributions and issues are welcome. Please follow standard GitHub workflows: open an issue to discuss major changes, then submit a pull request with tests and documentation updates.

## License

Copyright (c) 2026 Eduardo Morteo. All contents of this repository are
available under the [MIT License](LICENSE).

Contributors must not add photographs, research data, third-party assets, or
other material unless they have authority to distribute it under this license.
