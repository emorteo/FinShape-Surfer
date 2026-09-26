# GitHub Baseline Availability Review

Reviewed remote: `https://github.com/emorteo/FinShape3D`  
Reviewed commit: `cb091f3dc3ebcc81b7e4e389e39c201c41b4f5c9`  
Review date: 2026-09-23

## Finding

The remote repository did not contain everything needed to freeze the current
manual FinShape Surfer workflow.

| Required baseline item | Remote at reviewed commit | Local project evidence |
|---|---|---|
| FinShape Surfer 1.2.2 entry point | Missing | Available |
| Inherited 1.2.1, 1.2.0, and 1.1.3 sources | Missing | Available |
| FinShape2D 2.0.20 | Missing | Available |
| FinShape3D 1.1.20 | Missing | Available |
| CFD surface exporter 0.1.0 | Missing | Available |
| Standard and high-resolution texture modules | Missing | Available |
| Measurement example asset | Missing | Available |
| 1.2.x release hashes and validation evidence | Missing | Available |
| Tests for older modular prototype | Available | Available |
| Golden numerical manual-workflow fixture | Missing | Added in this working copy |
| Exact source-folder output-path regression | Missing | Added in this working copy |
| CI for the preserved current baseline | Missing | Added in this working copy |

The original remote tests are useful but do not establish a current baseline.
The old integration test checks only that two PNG paths exist; it does not
compare their geometry or numerical contents.

## Working-copy additions

- `baselines/FinShape_Surfer_1.2.2/`: exact release source chain and release
  evidence, preserved separately from the older root implementation.
- `tests/fixtures/surfer_1_2_2/manual_workflow_golden.json`: deterministic
  measurement, contour, reconstruction, and wireframe-reference baseline.
- `tests/regression/test_surfer_manual_baseline.py`: seven golden regression
  tests using explicit numerical tolerances and complete array hashes.
- `tools/generate_surfer_1_2_2_golden.py`: deliberate fixture regeneration.
- `.github/workflows/surfer-baseline.yml`: non-GUI regression automation.

## Execution results

| Command | Result |
|---|---|
| `python tests/regression/test_surfer_manual_baseline.py` | PASS: 7 |
| archived 1.2.2 non-GUI high-resolution suite | PASS: 8 |
| `pytest -q` for repository tests | PASS: 28 |
| native Tk use case inside sandbox | BLOCKED: macOS exit 134 |
| native Tk use case outside sandbox | BLOCKED: generated measurement/2D/3D/manifest outputs, then did not terminate and was interrupted |

The native Tk result is not reported as a pass. Its historical release evidence
records a successful run on the same project computer, but the current audit
requires a fresh complete result before claiming current GUI verification.

## Publication and data status

On 2026-09-25, copyright holder Eduardo Morteo confirmed that all current
repository contents are released under the MIT License. Future field
photographs, catalog data, validation artifacts, or third-party material must
not be added unless the contributor has authority to distribute them under the
same license and applicable research-data governance permits publication.
