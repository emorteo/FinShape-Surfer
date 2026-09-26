# FinShape Surfer 1.2.2 Release

Run the application with the project interpreter:

```bash
python Code/FinShape_Surfer_GUI_1.2.2.py
```

The original validation record used a project virtual environment. Its
machine-specific absolute interpreter path is redacted from this public copy.

Keep the release files together. `FinShape_Surfer_GUI_1.2.2.py` extends the preserved 1.2.1 application and loads `FinShapeTextureHQ_0.2.0.py` for the corrected high-resolution Photo-ID view.

## Changes

- Corrected Left to negative-Z and Right to positive-Z anatomical skin selection.
- Removed closed-mesh edge-wall faces from the high-resolution photograph only.
- Added 2x silhouette antialiasing and back-face culling without increasing mesh density.
- Matched the Photo-ID camera convention to the Matplotlib wireframe view.
- Preserved pitch, yaw, and roll across mode and side changes.
- Kept the mode-aware save button visible and added 1920 x 1440 high-resolution PNG saving.
- Recorded display method, side, camera, and unchanged-CFD status in JSON provenance.

Measurement, Bezier, morphometric, 2D, 3D wireframe, and simulation-surface behavior are inherited unchanged. See `Documentation/FinShape_Surfer_1.2.2_3D_Display_Corrections.md` and `Code/Surfer_validation_1.2.2/TEST_COMPLETION.md` for scope and evidence.

The high-resolution photo is an uncalibrated affine visualization for analyst inspection. It is not automated identity evidence and is not a CFD or thermoregulation result.
