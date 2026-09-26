# FinShape Surfer 1.2.2 Baseline Freeze

## Purpose

This repository originally contained an older modular prototype identified in
its source as FinShape2D 2.0.13 and FinShape3D 1.1.9. It did not contain the
tested FinShape Surfer 1.2.2 application available on the project computer.

The versioned directory `baselines/FinShape_Surfer_1.2.2/` preserves the exact
reviewed release without replacing or rewriting the older repository files.

## Authority

For regression purposes, the preserved 1.2.2 release is authoritative for the
manual workflow that existed before automatic computer-vision work:

1. load, flip, rotate, and crop an image;
2. place B, A, C30, C20, C10, C5, and D manually;
3. adjust the permitted Bezier controls;
4. save the measurement CSV;
5. generate the measured 2D contour and ratio-driven 2D reconstruction;
6. generate and inspect the 3D wireframe;
7. inspect the one-sided high-resolution photographic rendering;
8. export an explicitly experimental simulation surface.

The freeze does not assert that a formula is biologically valid merely because
the implementation is stable. It protects observed behavior so later changes
can be separated from intentional scientific revisions.

## Protected evidence

- `baselines/FinShape_Surfer_1.2.2/SHA256.json` protects release source,
  documentation, the measurement example, and archived output images.
- `tests/fixtures/surfer_1_2_2/manual_workflow_golden.json` records a synthetic
  manual landmark case, measurement row, Bezier roles and controls, full contour
  hashes, reconstructed landmarks/controls, curve hashes, area, edge references,
  and output-directory policy.
- `tests/regression/test_surfer_manual_baseline.py` recomputes those values from
  the preserved production code and compares them with explicit tolerances.
- The archived 1.2.2 display test is retained under the baseline directory. Its
  required 1.2.1 fixture helper is included so the test is self-contained.

The synthetic landmark fixture is a software regression case. It is not a
claim about a real dolphin, anatomical plausibility, or scientific accuracy.

## Running the freeze

The core freeze uses `unittest` and does not require pytest:

```bash
python tests/regression/test_surfer_manual_baseline.py
python baselines/FinShape_Surfer_1.2.2/Code/Surfer_validation_1.2.2/test_hq_view_fixes.py
```

The real Tk use case is opt-in because it opens native windows:

```bash
FINSHAPE_LIVE_TK=1 python \
  baselines/FinShape_Surfer_1.2.2/Code/Surfer_validation_1.2.2/test_live_tk_hq_fixes.py
```

## Updating the golden fixture

Do not update the golden fixture just to make a failing test pass. First decide
whether the difference is a defect, an intended software change, or an approved
scientific change. Record the decision and then run:

```bash
python tools/generate_surfer_1_2_2_golden.py
```

Review the resulting diff, run all regression tests, and retain the old fixture
under its version when behavior changes intentionally.

## Remaining gaps

- Native save-dialog clicking is not automated.
- A real field photograph is not committed as a generic regression fixture;
  future additions still require rights and research-data governance review.
- There is no calibrated camera or 3D ground truth for scientific registration.
- The regression freeze does not validate CFD or thermoregulation assumptions.
