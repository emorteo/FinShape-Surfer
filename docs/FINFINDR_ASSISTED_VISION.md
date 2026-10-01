# FinFindR-assisted vision

## Status

This feature is **experimental operator assistance**, not validated automatic
morphometry. It proposes a fin-centered crop and a complete photo contour. It
does not place landmarks, alter Bezier controls, change a saved measurement,
identify an animal, or replace the measured CFD boundary. It may draw
review-only crop and contour overlays in the image, 2D, and wireframe views,
and its reviewed contour is used to mask the Photo-ID display.

The frozen FinShape Surfer 1.2.2 release under `baselines/` is unchanged. The
existing manual workflow remains available when PyTorch or the external models
are absent.

## Evidence boundary

The integration is pinned to FinFindR commit
`5d7babd208af9642e440ef9143ec679c1154ff91`, whose `DESCRIPTION` declares
version 0.1.21. The upstream source and model files are GPL-2-or-later and are
not copied into this MIT-licensed repository.

The implementation uses:

- FinFindR's exact `cropperInit-0941` crop checkpoint;
- FinFindR's exact six-class
  `SWA_cont2_traceLong7_bn_6,10,5_RGB_fin-0000` edge checkpoint;
- a narrow, inference-only reader/executor for the MXNet operators present in
  those two pinned graphs;
- morphology and bounding-box rules ported from `R/blobber.R`; and
- a FinShape Surfer weighted A* adapter that combines the learned trailing,
  tip, and leading-edge maps with a higher-resolution image-gradient corridor;
  and
- a robust color-boundary term seeded by the operator-fitted Bezier interior
  and the outer crop frame, used only after A, B and C30 have been placed.

The A* adapter is not represented as identical to FinFindR's full
`traceFromImage()` post-processing. Original R/MXNet numerical parity remains
unverified because the legacy R MXNet runtime is unavailable on the tested
Apple-silicon environment. Model execution, deterministic adapter outputs, and
one private local use-case preview have been tested; the private photograph is
not committed.

The network still operates at a 200-pixel dominant dimension, as in FinFindR.
For an operator-anchored final path, its probabilities first define a bounded
corridor. Full-resolution RGB gradients and the robust tissue/water color model
then define the pixel-level route. This preserves source-image notch detail
without making the Tk event loop wait for inference.

## Geometry and display contract

The application deliberately keeps three products separate:

1. **Measured morphometry:** landmarks, Bezier controls, CSV values, the
   measured 2D/wireframe representation, and the CFD boundary. These remain
   the scientific measurement authority.
2. **Review overlays:** the green crop proposal and orange B-to-A-to-C30
   contour. They are operator feedback and are never silently committed as
   landmarks or controls.
3. **Photo-ID display:** the smooth measured B-to-D-to-A leading boundary is
   combined with the reviewed detailed A-to-C30 contour to mask the photographed
   skin. This removes water outside the identity-bearing trailing edge while
   retaining visible nicks and notches. It is display geometry, not a
   replacement CFD surface.

## Installation

Install FinShape Surfer with the complete GUI dependency profile:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[full]'
```

Fetch the exact external checkout into a user cache and verify all four model
file hashes:

```bash
python tools/prepare_finfindr_models.py
```

Set `FINSHAPE_FINFINR_MODEL_DIR` to the directory printed by that command. The
GUI automatically checks the preparation tool's default cache. Set
`FINSHAPE_FINFINR_MODEL_DIR` only when using a different cache location. The
GUI never downloads software or models silently, and every discovered model
file is verified against its pinned SHA-256 value before use.

## GUI behavior

Launch the active GUI with `python finshape_surfer_gui.py`. It subclasses the
frozen 1.2.2 release without changing those baseline files. The **Assisted
vision** row contains three unnumbered experimental actions so the established
manual workflow numbering is not changed.

1. **Automatic fin detection** runs in a background thread. FinFindR's crop network
   first supplies a coarse dolphin location; its tip, leading-edge and
   trailing-edge classes then refine that location to a fin-centered green crop.
   A tiled fallback helps when the fin is small in a high-resolution frame.
   The operator sees it before using the separate **Apply detected crop**
   action. Manual cropping and crop reset clear the proposal. Because the fin
   can join the back gradually, this crop is deliberately a review aid and is
   not claimed to identify a biological fin-base boundary.
2. **Automatic edge detection** becomes available after A, B and C30
   exist. It draws a complete orange B-to-A-to-C30 path. Those landmarks are
   authoritative endpoints; FinFindR's leading, tip and trailing classes,
   source-resolution gradients, and robust tissue/water color evidence propose
   the intervening paths. The thin line never moves a landmark or Bezier control.

The fitted C5-to-C20 segment remains exactly anchored at C10. From version
1.3.5 onward its green control is unrestricted: it controls the first of two
quadratic segments, and a reflected second control gives a continuous tangent
through fixed C10. The operator can therefore drag to either side of the
visible curve without making the C10 landmark decorative or movable. An exact
de Casteljau subdivision preserves the inherited curve until that handle is
actually dragged.

The B-to-D leading segment remains a cubic Bezier with two independent control
points. Version 1.3.4 makes both controls and their endpoint links explicit so
the operator can shape the segment near B and near D without changing either
fixed landmark.

If an image transform changes while inference is running, the stale result is
discarded. Loading another image, flipping, rotating, resetting a crop, or
applying a crop clears previous suggestions.

After a measurement is saved, automatic edge detection remains available as a
read-only review action. Closing the derived 2D window restores the main window
and its review controls; a user does not have to reload the photograph or repeat
landmark placement.

The orange trailing segment remains visible in the image workspace, 2D contour
window and 3D wireframe as a review layer. Photo-ID rendering combines that
learned A-to-C30 segment with the fitted B-to-D-to-A leading Beziers. It does
not replace the fitted CFD boundary, alter exported CFD geometry, or count as
an accepted measurement.

## 1.3.8 review and reconstruction fixes

- Photo-ID reconstruction uses the exact measured B and C30 endpoints for its
  attachment row. Interior rows continue to use checked polygon intersections.
  This avoids the GEOS boundary-intersection ambiguity that could report zero
  horizontal chords at `Y=0` for an otherwise valid planform.
- The assisted-vision row credits `FinFindR 0.1.21, Jaime Thompson` directly.
- Closing the 2D view explicitly clears its active state, restores the main
  window, and recomputes review-action availability.

## 1.3.11 rendered-regression correction

- The endpoint-only 1.3.10 wire test was insufficient: it permitted a center
  rail to touch the singular apex, which remained visibly triangular. Every
  longitudinal display rail now terminates on one common cap section. Sparse
  transverse rings describe the measured tip while the black measured outline
  remains authoritative through the apex. CFD geometry is unchanged.
- The dense structured 321-by-41 Photo-ID mesh from 1.3.8 is restored whenever
  every horizontal section is one connected tissue chord. This prevents coarse
  arbitrary triangles from becoming a stair-stepped silhouette at oblique
  views. The 1.3.10 boundary-conforming lift is retained only for planforms
  that genuinely contain multiple tissue intervals at one height.
- Tests now render the actual Matplotlib wireframe and assert that no blue
  artist or raster pixel reaches the projected apex. A second raster test
  renders the structured Photo-ID surface at yaw -159 degrees and checks both
  silhouette edges for high-frequency steps. The reported saved measurement
  is also replayed manually at its reported pose before release.
- Image, morphometry, and FinFindR credit labels are right-aligned and use the
  same default toolbar font.

## 1.3.7 display reconstruction

- Version 1.3.6 did not fix the two release-blocking defects. Its separate cap
  made the edge-on tip concave, while its textured ribbon concealed rather
  than replaced the faceted leading surface and could appear as a duplicate
  edge. Both mechanisms have been removed.
- Wireframe and Photo-ID views now use the same closed symmetric NACA-style
  section law. Maximum full thickness is calibrated from the inherited model,
  limited to 12 percent of each local chord, and forced to zero at both contour
  edges and at the apex.
- Photo-ID is rebuilt as one structured span/chord mesh with 321 span stations
  and 41 chord stations. This conforming surface is the photographed skin and
  the silhouette; there is no ribbon, cap, duplicate edge, or other auxiliary
  surface.
- The measured planform remains authoritative. This reconstructed thickness is
  a documented display assumption, not measured biological thickness. The
  experimental CFD exporter is deliberately unchanged and is not represented
  as validated by this Photo-ID correction.

The closed-section formulation follows the standard analytic thickness-distribution
practice documented for NACA airfoils, including explicit leading-edge shape
and continuous closure: [NASA TM-4741](https://ntrs.nasa.gov/citations/19970008124)
and [NACA Report 824](https://ntrs.nasa.gov/citations/19930090976).

## CFD geometry decision

The experimental CFD exporter and the airfoil-style display reconstruction have
different requirements and remain intentionally separate:

- The exporter preserves every measured Bezier boundary sample, creates a
  constrained planar triangulation, lifts matching positive and negative skins,
  closes all boundary walls, and rejects surfaces that are not watertight,
  consistently wound, connected, positive-volume genus-zero solids.
- The display reconstruction provides better visual section continuity, but its
  positive and negative skins meet through coincident zero-thickness edge and
  apex vertices. That is appropriate for rendering and inappropriate to
  substitute directly as a CFD boundary: coincident vertices do not by
  themselves form a solver-safe manifold.
- Therefore the validated original export construction is the better current
  CFD input path. This is a topology decision, not a claim that its empirical
  thickness assumption is biologically validated. Replacing that thickness law
  requires measured thickness data or an explicit scenario plus a dedicated
  watertight mesh generator, sensitivity analysis, and solver validation.

## Rejected 1.3.6 approach

- The 1.3.5 approach did not resolve the two remaining display defects on the
  reported real cases: stopping rails below the apex left the tip unwired, and
  retaining coarse photographed-skin boundary triangles still exposed a large
  staircase at oblique yaw.
- The wireframe reserved a finite-width shoulder and drew a separate
  rounded cap grid. Every longitudinal cap rail ends at a distinct arclength
  position on the measured tip arc, and transverse rings continue through the
  cap. This is display geometry only; the surface mesh and CFD triangulation
  are unchanged.
- Photo-ID mode replaced only a narrow leading-edge display band with a dense
  analytic textured ribbon following the fitted B-to-D-to-A Bezier edge. The
  ribbon tapers to zero at B and A and overlaps the retained photographed skin,
  preventing endpoint tabs, seams, and coarse-triangle staircases.

Those mechanisms are no longer present. The remaining assumptions inherited
from 1.3.5 are:

- Interactive wireframe and Photo-ID displays calibrate maximum thickness from
  one half of the inherited estimate, then enforce a 12-percent local-chord
  limit and continuous edge/apex closure. The experimental CFD exporter is
  unchanged.
- Photo mode renders only the operator-selected photographed skin and keeps it
  depth-tested at tangent angles. Hiding the opposite skin and edge wall avoids
  false triangle-shaped occlusions along the smooth measured Bezier silhouette.
- For a dolphin swimming to the right, the anatomical display convention is
  Left = +Z and Right = -Z.
- Detected crop and edge overlays are cleared from both application state and
  the canvas whenever crop state or image transforms change.
- The high-resolution photograph uses a hybrid display boundary: the smooth,
  operator-fitted cubic B-to-D-to-A leading edge and the reviewed dense
  FinFindR A-to-C30 trailing edge. This removes learned source-pixel stairs
  from the streamlined edge while retaining nicks, notches, and water removal
  on the identity-bearing trailing edge. CFD export remains measured geometry.
- Moving the 2D rotation handle right means clockwise. Pitch and yaw slider
  directions follow the displayed fin, and dragging the 3D view directly
  changes yaw and pitch; Shift-drag changes roll.
- Axis labels and tick text are constrained to remain no larger than the 3D
  title, and the axes occupy the available plotting canvas.
- Photo-ID leading-edge geometry comes from the measured cubic Beziers rather
  than a raster path. Exact B, D, and A landmarks and both manual controls are
  therefore respected. The A-to-C30 learned path is not geometrically smoothed,
  so nicks and notches remain visible. This is a display-processing choice,
  not inferred anatomy.
- The wireframe displays at most 15 uniformly spaced chordwise rails and an
  even body grid. All rails and transverse sections use the same continuous
  section law and terminate at the zero-thickness measured apex.
- The cited measurement example is cleaned only for display to hide an
  unrelated upper-right scan artifact. Text corrections are performed in place
  so the embedded example is retained. The frozen source asset is unchanged.

## 1.3.13 convex tip closure and responsive camera preview

- The ruled tip keeps the 1.3.12 equal-arclength planform targets and shared
  longitudinal/transverse topology. Its thickness now closes with a
  quarter-elliptic envelope, which is tangent to the body thickness and convex
  in end view while still reaching exact zero thickness at the measured tip
  contour. This is display geometry only; measurement and CFD geometry do not
  change.
- Active Pitch, Yaw, Roll, and direct-drag motion uses a cached,
  boundary-preserving display mesh with a 120,000-pixel, single-sample Photo-ID
  preview. The reduced preview preserves the reviewed boundary while using
  fewer span/chord stations so camera controls remain responsive. Control
  release requests the unchanged full mesh at 420,000 pixels and 2x
  antialiasing. The 1920 x 1440 saved image path remains full resolution, full
  mesh, and 2x antialiased.

## 1.3.12 tip topology and color-boundary correction

- The 1.3.11 apex-absence test was not a valid test of a closed tip: it could
  pass while the upper wire families were simply missing. The wireframe now
  constructs one ruled tip patch from the last body chord to the measured tip
  arc. Every longitudinal rail has a different equal-arclength contour target,
  and every transverse ring is sampled from the same parameterization. Tests
  verify exact rail/ring intersections, distinct contour endpoints, zero
  contour thickness, and containment within the measured cap.
- Legacy constant-thickness curves that ended inside the fin are omitted from
  the display. Complete body sections and the new cap grid remain; measured
  geometry and CFD export are unchanged.
- Genuine multi-interval Photo-ID masks now receive a dense constrained-
  Delaunay triangulation. Every measured concave contour segment is a mesh edge;
  shared-edge refinement cannot bridge a notch or stretch a few coarse image
  triangles across the fin. This removes the faceted oblique staircase while
  preserving the reviewed boundary.
- Source-resolution contour refinement now evaluates robust CIE Lab color and
  chroma separation between sampled tissue and water, in addition to edge
  probability and local gradient. A regression fixture uses equal-luminance
  dark blue and dark gray regions so a luminance-only implementation cannot
  pass it.

## Auditable command-line preview

The standalone preview is useful before enabling the GUI feature:

```bash
python tools/finfindr_assist.py path/to/image.jpg \
  --preview outputs/review.png \
  --metadata outputs/review.json
```

When a crop is found, the command traces the edge inside that crop and maps the
path back to source-image coordinates. The JSON records the pinned commit,
model hashes, thresholds, output coordinates, warnings, and the mandatory
manual-review status. It does not alter the input image.

## Validation and limitations

- All original repository tests and frozen manual-workflow hash checks must
  pass before merging.
- External checkpoint tests run only when `FINSHAPE_FINFINR_MODEL_DIR` is set.
- The learned checkpoints may fail on unfamiliar image domains, occlusion,
  glare, poor crops, unusual orientation, or partial emergence.
- Detection confidence is a model/post-processing score, not a calibrated
  biological probability.
- The edge suggestion must not be used to invent missing landmarks.
- Automatic landmark placement requires a separate labeled-data protocol,
  validation study, error tolerances, and explicit operator acceptance.

On the development Apple M1, a private 3456 x 2304 JPEG use case produced one
crop and a 299-point edge suggestion in about 1.6 seconds on the first run and
1.1 seconds after model warm-up. These are observations from one image, not a
general performance guarantee. The photograph and its identifiers are not
part of the repository.

## Verification record

Verification completed on 2026-10-01 for version 1.3.13:

- all 91 automated tests passed with the pinned FinFindR 0.1.21 checkpoint
  directory enabled, including frozen 1.2.2 workflow checks;
- the reported EN76 measurement was reconstructed from its saved CSV and
  rendered at neutral Pitch, Yaw, and Roll. The front-view cap is convex, every
  rail retains a distinct measured contour endpoint, and the shared transverse
  rings remain closed;
- tests verify the quarter-ellipse cap equation, exact zero-thickness contour
  closure, low-latency motion budget, unchanged released preview settings, and
  unchanged 1920 x 1440 export settings; and
- focused lint, Python compilation, dependency consistency, and whitespace
  checks passed.

Verification completed on 2026-10-01 for version 1.3.12:

- all 91 automated tests passed with the pinned FinFindR 0.1.21 checkpoint
  directory enabled, including the frozen 1.2.2 baseline checks;
- the wireframe replay from the reported EN85 measurement contains one closed
  ruled tip grid with distinct equal-arclength contour endpoints, exact
  longitudinal/transverse intersections, and no incomplete constant-thickness
  diagonals;
- the multi-interval stress fixture verifies that every measured concave
  contour segment survives as a connected mesh-edge path and that an edge-on
  raster has no high-frequency staircase;
- the reported EN85 photograph was replayed without modification using its
  saved crop and operator landmarks. The Lab-assisted trace followed the two
  dark-blue-water versus dark-gray-tissue notch transitions and retained exact
  A and C30 endpoints; and
- focused lint, Python compilation, dependency consistency, and whitespace
  checks passed.

Verification completed on 2026-09-30:

- 83 automated tests passed with the pinned model directory enabled, including
  all pre-existing tests and frozen baseline hash checks;
- all four external checkpoint files matched their pinned SHA-256 values;
- the deterministic R/C++ reference golden verifier passed;
- Python compilation, dependency consistency, whitespace, and repository
  secret/path scans passed;
- the 1.3.8 GUI initialized successfully on macOS while retaining the frozen
  manual 2D and 3D sources; and
- the fin-centered refinement was exercised on reported private photographs,
  and the source-pixel A-to-C30 trace retained visible notch detail without
  recording source paths or modifying either photograph;
- the learned B-to-A-to-C30 contour is independently recorded as a Photo-ID-only
  mask while measured Bezier and CFD snapshots remain unchanged;
  and
- every measured Bezier boundary sample is verified to survive constrained CFD
  triangulation exactly, so the display-only tangent-view repair cannot change
  the exported simulation surface.

The reported release-blocking reconstruction was replayed from its saved
measurement at pitch 1 degree and yaw 0, 6, and -40 degrees, plus an oblique
wireframe pose. Photo-ID used one structured mesh containing 26,242 vertices
and 51,200 triangles; the selected photographed skin contained 25,560 of those
triangles. Each pose produced one connected silhouette with no auxiliary
surface, ribbon, cap, or duplicate edge. The wireframe contained 167 lines,
with longitudinal and transverse lines continuing through a convex,
continuously tapering tip to one zero-thickness apex. Photo-ID frame times were
approximately 2.1 to 2.6 seconds at 1176 x 681 pixels with 2x antialiasing on
the development Apple M1. The private photograph and its identifiers are not
stored in the repository.

The verification does not establish scientific validity, field-domain
accuracy, identity-matching performance, or numerical parity with legacy R
MXNet. Those remain separate validation requirements.
