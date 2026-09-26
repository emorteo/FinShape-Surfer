# FinShape Surfer 1.2.2 Test Completion

Final execution: 2026-09-11.

Interpreter: project virtual environment on the project's Apple M1 computer.
The machine-specific absolute interpreter path is redacted from this public
copy; the recorded results and timings are unchanged.

## Final results

| Validation | Result |
| --- | --- |
| 1.2.2 non-GUI unit/regression suite | PASS: 8; FAIL: 0; SKIP: 1 opt-in real-Tk case |
| 1.2.2 real-Tk end-to-end use case | PASS: 1; FAIL: 0 |
| Preserved 1.2.1 suite, standalone | PASS: 12; FAIL: 0; SKIP: 1 opt-in real-Tk case |
| Preserved 1.2.0 suite | PASS: 11; FAIL: 0; SKIP: 1 opt-in real-Tk case |
| Preserved 1.1.3 baseline suite | PASS: 31; FAIL: 0 |
| Preserved 1.1.3 Surfer suite, non-display | PASS: 87; FAIL: 0; live-display skips: 27 |
| Project-interpreter compilation | PASS |
| `pip check` | PASS: no broken requirements |
| Protected-source SHA-256 comparison | PASS: 8 of 8 exact |

## What the tests prove

- Left selects the negative-Z skin and Right selects the positive-Z skin.
- Each corrected side is visible from its corresponding broad-side camera position.
- `edge_wall` faces produce no pixels in high-resolution photo mode.
- 2x supersampling adds antialiased silhouette colors and retains output dimensions.
- The software camera uses the same azimuth meaning as Matplotlib.
- Switching wireframe/photo mode and anatomical side retains pitch, yaw, and roll.
- The high-resolution save button stays packed and visible, names its current action, and saves a 1920 x 1440 PNG to the selected source directory.
- The real-Tk workflow performs Save Measurement, Show 3D, both side choices, both view modes, camera retention, and a high-resolution save.
- A synthetic 1000 x 700 antialiased frame stays below the conservative 3.0-second M1 regression ceiling; the measured real-Tk interactive frame stayed below two seconds.
- New classes do not override measurement, landmark, Bezier, integrated-output, or CFD-export methods.
- Eight protected predecessor sources retain their expected SHA-256 digests.

## Retained unsuccessful evidence

- The first new side-visibility test exposed an inverted front-face culling sign. It failed and was corrected before final validation.
- Early real-Tk runs passed but took 48.37 and 42.86 seconds end to end while scene refinement and repeated construction were enabled. The final implementation reuses immutable geometry and does not refine the display mesh; the complete final use case took 15.23 seconds, including measurement output, Matplotlib construction, view switching, both sides, and the full-size save.
- Running four historical suites concurrently caused the old 1.2.1 five-second save benchmark to take 6.75 seconds. The same preserved suite passed when rerun alone. This resource-contention result is not represented as a product regression.
- A final 1.2.2 suite run produced a 2.52-second synthetic frame and failed the original brittle 1.5-second gate; an immediate standalone repeat passed. The regression ceiling was conservatively set to 3.0 seconds while preserving the measured timings above.
## Remaining limits

- Native macOS save-dialog clicking is not automated; the real application save boundary, output path, image dimensions, and file creation are tested.
- No calibrated camera or 3D ground truth is available for quantitative registration validation.
- High-resolution display does not validate identity, biological thickness, simulation readiness, or thermoregulation.
