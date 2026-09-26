# FinShape Surfer 1.2.2

FinShape Surfer is research software for manually measuring bottlenose-dolphin
dorsal-fin photographs and inspecting corresponding 2D and modeled 3D
representations. This directory preserves version 1.2.2 as a regression
baseline before automated fin detection, edge extraction, or landmark placement
is introduced.

> [!IMPORTANT]
> This is a frozen behavioral baseline, not the repository's current
> development area. Do not edit these versioned files in place. Intentional
> behavioral changes should be released under a new version and accompanied by
> reviewed regression evidence.

## Project status

| Item | Status |
|---|---|
| Manual image measurement | Implemented and regression-protected |
| Measured and ratio-driven 2D contours | Implemented and regression-protected |
| Rotatable 3D wireframe | Implemented and regression-protected |
| One-sided photographic 3D display | Implemented for analyst inspection |
| Experimental closed-surface export | Implemented; not validated as CFD-ready |
| Automated fin or edge detection | Not implemented |
| Automated landmark placement | Not implemented |
| Automated individual identification | Not implemented |
| CFD or bioheat simulation execution | Not implemented |

The photographic view is an uncalibrated affine visualization. It does not
perform camera calibration, lens correction, pose recovery, or parallax
correction, and it must not be interpreted as automated identity evidence.

## Features

- Load, flip, rotate, and crop supported dorsal-fin photographs.
- Place landmarks manually in the order `B -> A -> C30 -> C20 -> C10 -> C5 -> D`.
- Adjust only the permitted Bezier control points.
- Save measurements and derived morphometric ratios to CSV.
- Export measured contours as SVG and, when Pillow is available, PNG, JPEG, or
  BMP.
- Generate measured and reconstructed 2D views and a rotatable 3D wireframe.
- Inspect a one-sided, high-resolution photographic rendering while retaining
  pitch, yaw, and roll between display modes.
- Save 3D views and a 1920 x 1440 high-resolution photographic PNG with JSON
  display provenance.
- Export an experimental closed-surface package containing STL, OBJ, indexed
  mesh data, geometric patch groups, validation results, provenance, and an
  incomplete simulation template.

## Requirements

The preserved release was developed for Python 3.12 with Tkinter and validated
on macOS. Other Python versions and operating systems are currently
**unverified**.

Required Python packages:

- NumPy
- pandas
- Matplotlib
- Pillow
- SciPy
- openpyxl

Optional surface-export packages are pinned in
[`Code/requirements-cfd-0.1.0.txt`](Code/requirements-cfd-0.1.0.txt):

- Shapely 2.1.2 with GEOS 3.10 or newer
- trimesh 4.11.2

Tkinter normally ships with the official macOS and Windows Python installers.
On Linux, install the distribution-specific Tk package separately when it is
not included with Python.

## Installation

Clone the repository and create an isolated environment from the repository
root:

```bash
git clone https://github.com/emorteo/FinShape-Surfer.git
cd FinShape-Surfer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib Pillow scipy openpyxl
python -m pip install -r baselines/FinShape_Surfer_1.2.2/Code/requirements-cfd-0.1.0.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Keep every file under `Code/` together. The 1.2.2 entry point deliberately
loads its preserved 1.2.1, 1.2.0, and 1.1.3 predecessors and the versioned
geometry and rendering modules by relative path.

## Running the application

From the repository root:

```bash
python baselines/FinShape_Surfer_1.2.2/Code/FinShape_Surfer_GUI_1.2.2.py
```

Basic workflow:

1. Load a photograph and orient the animal so it is swimming to the right
   before landmarking.
2. Apply any required rotation or crop.
3. Place `B`, `A`, `C30`, `C20`, `C10`, `C5`, and `D` in that order.
4. Review and, where permitted, adjust the Bezier controls.
5. Save the measurement CSV.
6. Inspect the measured 2D contour, reconstructed 2D view, 3D wireframe, and
   high-resolution photographic view.
7. Review all warnings and provenance before using or publishing an output.

Image interpretation and landmark placement remain operator-controlled. Missing
landmarks are not automatically estimated by this release.

## Outputs

The application uses the source photograph's directory as the default save
location. Integrated outputs are written to a versioned sibling directory and
include, as applicable:

- measurement CSV;
- measured contour SVG;
- measured or reconstructed 2D PNG;
- 3D wireframe PNG;
- high-resolution photographic PNG and JSON provenance;
- text provenance manifest; and
- an optional experimental simulation-surface archive.

The surface archive includes `surface.stl`, `surface.obj`, `mesh.npz`,
`patches.json`, `validation.json`, `input.json`, `manifest.json`, a preview, and
an incomplete `simulation_setup.json`. STL coordinates are expressed in metres,
but STL itself stores no unit metadata; the receiving application must be
configured explicitly.

Generated manifests and surface packages may contain absolute source paths and
catalog identifiers for reproducibility. Inspect and, where appropriate,
sanitize those records before sharing an output outside the research project.

> [!WARNING]
> The exported surface is not a volume mesh or a runnable CFD or
> thermoregulation model. Modeled thickness, scale, boundary conditions,
> material properties, perfusion assumptions, and mesh suitability require
> independent scientific validation.

## Testing the frozen baseline

Install the test runner if it is not already available:

```bash
python -m pip install pytest
```

Run the repository and golden regression suites from the repository root:

```bash
MPLBACKEND=Agg python -m pytest -q
```

The baseline can also be checked directly without pytest:

```bash
python tests/regression/test_surfer_manual_baseline.py
python baselines/FinShape_Surfer_1.2.2/Code/Surfer_validation_1.2.2/test_hq_view_fixes.py
```

The native Tkinter use case opens application windows and is opt-in:

```bash
FINSHAPE_LIVE_TK=1 python \
  baselines/FinShape_Surfer_1.2.2/Code/Surfer_validation_1.2.2/test_live_tk_hq_fixes.py
```

The non-interactive suites protect numerical measurements, Weller ratios,
Bezier roles and controls, contour orientation and normalization, reconstructed
landmarks and curves, wireframe references, output-directory behavior, and the
1.2.2 display extension's isolation from inherited scientific behavior. See
[`../../docs/BASELINE_FREEZE_1.2.2.md`](../../docs/BASELINE_FREEZE_1.2.2.md) for
the exact scope and known test gaps.

## Baseline integrity

[`SHA256.json`](SHA256.json) records hashes for the preserved source,
documentation, assets, and archived validation evidence. The golden fixture is
stored at
[`../../tests/fixtures/surfer_1_2_2/manual_workflow_golden.json`](../../tests/fixtures/surfer_1_2_2/manual_workflow_golden.json).

Do not regenerate a golden fixture merely to silence a failing test. First
classify the difference as a defect, an intentional software change, or an
approved scientific change. Preserve the previous version and document the
decision before accepting new expected values.

## Repository layout

```text
baselines/FinShape_Surfer_1.2.2/
|-- Code/                   Versioned application and scientific modules
|   |-- assets/            In-application measurement example
|   `-- Surfer_validation_1.2.2/
|                           Archived release tests and evidence
|-- Documentation/         Version-specific technical notes
|-- README.md               This guide
`-- SHA256.json             Frozen-file integrity manifest
```

## Scientific limitations

Stable software behavior is not equivalent to scientific validation.

- The golden landmark fixture is a software regression case, not evidence of
  anatomical plausibility or population-level validity.
- Photograph warping may help an analyst inspect viewpoint effects, but it does
  not prove that two photographs show the same individual.
- Modeled thickness is a documented geometric assumption, not directly measured
  anatomy.
- Successful surface generation or geometric validation does not establish CFD
  readiness.
- Surface area or CFD output alone cannot establish thermoregulatory function or
  evolutionary adaptation.
- Canonical reconstructed morphology and an observed individual fin must remain
  distinguishable in downstream metadata and publications.

The measurement example follows the landmark scheme described in:

> Morteo E, Rocha-Olivares A, Morteo R, Weller DW. 2017. Phenotypic variation
> in dorsal fin morphology of coastal bottlenose dolphins (*Tursiops truncatus*)
> off Mexico. *PeerJ* 5:e3415. <https://doi.org/10.7717/peerj.3415>

## Research data and privacy

Dorsal-fin photographs and generated outputs may contain catalog identifiers,
study-location information, unpublished observations, image metadata, or other
restricted research data. Do not commit field photographs or catalog outputs
unless their ownership, consent, institutional requirements, and publication
rights permit public redistribution.

The copyright holder has released the current repository contents, including
the synthetic validation artifacts, under the MIT License. Private catalog
artifacts and identifiers are excluded from the public repository. This does
not grant permission to add future field photographs, catalog data, or
third-party material unless the contributor has authority to release those
additions under the same license.

## Contributing

Please read [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) before proposing a
change. For baseline-sensitive contributions:

1. Open an issue describing the observed behavior and whether it affects
   software, geometry, scientific interpretation, or downstream fitness.
2. Keep changes focused and place new behavior in a new version rather than
   editing the frozen release in place.
3. Add or update tests without weakening existing tolerances or constraints.
4. Document scientific assumptions, input units, coordinate conventions, and
   provenance.
5. Include the commands and evidence used to validate the change.

Never include confidential photographs, animal-location data, credentials, or
other restricted material in issues or pull requests.

## Citation

A project-level `CITATION.cff` has not yet been provided. Until one is added,
cite the exact software version, repository URL, access date, and relevant
scientific method publication. Do not attribute scientific validation to this
software beyond the evidence reported for the cited version.

## License

Copyright (c) 2026 Eduardo Morteo. All contents of this repository, including
this preserved baseline, are available under the
[MIT License](../../LICENSE).

Contributors must not add photographs, research data, third-party assets, or
other material unless they have authority to distribute it under this license.

## Acknowledgments

FinShape Surfer builds on the project's FinShape2D measurement workflow,
FinShape3D reconstruction code, experimental surface exporter, and photographic
rendering modules. The source files and provenance manifests identify the exact
component versions used by this baseline.
