# FinShape Surfer 1.2.2: 3D Photo-ID Display Corrections

## Scope

FinShape Surfer 1.2.2 is an additive display release based on the preserved and tested 1.2.1 application. It corrects the high-resolution Photo-ID view and its controls. It does not change landmark acquisition, Bezier curves, morphometric calculations, measured 2D output, wireframe geometry, modeled thickness, or experimental simulation-surface export.

## Corrections

### Anatomical side selection

The earlier software mapped the **Left** button to the positive-Z skin and **Right** to the negative-Z skin. Version 1.2.2 uses explicit anatomical mappings:

| Button | Textured geometric skin |
| --- | --- |
| Left | negative Z (`-Z`) |
| Right | positive Z (`+Z`) |

Image flipping remains independent metadata. Flipping an image for measurement does not silently determine which anatomical side was photographed.

### High-resolution edge display

The black boundary visible in the high-resolution view came from the closed mesh's `edge_wall` faces. Those faces remain part of the simulation surface and wireframe, but 1.2.2 omits them from the photographic rendering. This removes the dark strip that obscured nicks and notches without modifying CFD geometry.

### Leading-edge and apex smoothing

The high-resolution renderer now uses two-times supersampling followed by Lanczos downsampling. This smooths projected triangle boundaries and the fin silhouette while retaining the existing triangle topology. Back-face culling avoids rasterizing hidden triangles, offsetting much of the extra antialiasing work.

This is display antialiasing, not biological smoothing. It does not alter measured contours, add landmarks, move nicks or notches, or claim additional source-image detail.

### Consistent camera controls

The software renderer now follows Matplotlib's axis order and elevation/azimuth convention. Wireframe mode uses an orthographic projection compatible with the high-resolution view. Switching view mode or anatomical side retains the current pitch, yaw, and roll values; side selection no longer changes the camera.

### High-resolution save control

The save control is packed before the long output-path label so it cannot be pushed out of view. Its label changes with the selected mode:

- `Save wireframe view...`
- `Save high-resolution photo...`

High-resolution photographs are saved as 1920 x 1440 PNG images with 2x supersampling. The inherited source-directory dialog is used, and a JSON provenance record identifies the anatomical side, geometric skin, camera angles, antialiasing method, renderer version, and unchanged CFD status.

## Validation limits

- The photograph is affinely draped from the measured 2D registration; there is no camera calibration, lens correction, pose recovery, or parallax correction.
- The anatomical side is selected by the operator and is not inferred from image content.
- Antialiasing improves the displayed silhouette but cannot recover photographic information absent from the source image.
- The photographic display does not validate individual identity, biological thickness, CFD readiness, or thermoregulatory behavior.
- Rendering time depends on window size, visible surface area, triangle count, and concurrent system load.

## Relevant implementation

- `Code/FinShape_Surfer_GUI_1.2.2.py`: versioned GUI extension, retained view state, mode-aware save control, orthographic wireframe, output metadata.
- `Code/FinShapeTextureHQ_0.2.0.py`: corrected side mapping, Matplotlib-compatible camera projection, back-face culling, edge-wall suppression, and supersampling.
- `Code/Surfer_validation_1.2.2/`: unit, regression, real-Tk use-case, performance, and protected-file hash evidence.
