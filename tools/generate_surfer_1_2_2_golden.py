#!/usr/bin/env python3
"""Generate the reviewed FinShape Surfer 1.2.2 manual-workflow fixture.

This script is intentionally not run by the test suite. Updating the golden
fixture is a reviewed scientific-baseline change, not an automatic repair.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baselines" / "FinShape_Surfer_1.2.2"
CODE = BASELINE / "Code"
OUTPUT = ROOT / "tests" / "fixtures" / "surfer_1_2_2" / "manual_workflow_golden.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def array_sha256(values, decimals: int = 12) -> str:
    canonical = np.round(np.asarray(values, dtype="<f8"), decimals=decimals)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def point_map(values):
    return {name: [float(coord) for coord in point] for name, point in values.items()}


def build_fixture() -> dict:
    fin2d = load_module("finshape_2d_2_0_20_golden", CODE / "FinShape2D_R-GUI_2.0.20.py")
    fin3d = load_module("finshape_3d_1_1_20_golden", CODE / "FinShape3D_1.1.20.py")
    integrated = load_module("finshape_surfer_1_1_3_golden", CODE / "FinShape_Surfer_GUI_1.1.3.py")

    landmarks = fin2d.Landmarks(
        B=(145.84028605482717, 89.99880810488676),
        A=(22.259833134684147, 12.856972586412395),
        C30=(28.208581644815254, 85.96901072705602),
        C20=(33.581644815256254, 66.20381406436233),
        C10=(30.895113230035754, 43.36829558998808),
        C5=(24.562574493444576, 28.592371871275326),
        D=(67.35518474374256, 24.178784266984504),
    )
    controls = fin2d.build_initial_controls(landmarks)
    ratios = fin2d.compute_weller_ratios(landmarks)

    measured_outline = fin2d.outline_polygon(controls, n_per_seg=900)
    display_outline = fin2d.outline_polygon(controls, n_per_seg=1100)
    standardized = fin2d.standardize_contour_units(
        display_outline, landmarks.B, landmarks.A, landmarks.C30, ab_units=10.0
    )
    area_px2 = fin2d.shoelace_area(measured_outline)

    def rel(point):
        return point[0] - landmarks.B[0], point[1] - landmarks.B[1]

    measurement = {
        "image_name": "manual_fixture.png",
        "AB_px": f"{ratios['AB_px']:.6f}",
        "area_px2": f"{area_px2:.6f}",
        "C5B_over_AB": f"{ratios['C5B_over_AB']:.10f}",
        "C10B_over_AB": f"{ratios['C10B_over_AB']:.10f}",
        "C20B_over_AB": f"{ratios['C20B_over_AB']:.10f}",
        "C30B_over_AB": f"{ratios['C30B_over_AB']:.10f}",
        "DB_over_AB": f"{ratios['DB_over_AB']:.10f}",
    }
    for name in ("A", "C5", "C10", "C20", "C30", "D"):
        x, y = rel(getattr(landmarks, name))
        measurement[f"{name}_dx"] = f"{x:.6f}"
        measurement[f"{name}_dy"] = f"{y:.6f}"

    row = integrated.measurement_row_to_3d_series(measurement, fin3d, "manual_fixture")
    warnings = []
    landmarks_3d, u_ab = fin3d.build_landmarks(row, BC30_scale=10.0)
    controls_3d = fin3d.build_controls(landmarks_3d, u_ab, warnings=warnings)
    curves_3d = fin3d.compute_2d_curves(controls_3d, n=900)
    y_max = float(curves_3d["outline"][:, 1].max())
    y_values = fin3d.contour_reference_y_values(
        curves_3d, y_max, n_y_slices=39, top_focus_power=3.0,
        n_edge_points=100, max_sections=55,
    )
    edge_refs = fin3d.contour_reference_points(curves_3d, n_edge_points=100)

    source_hashes = json.loads((BASELINE / "SHA256.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "baseline": "FinShape Surfer 1.2.2",
        "source_release_manifest_sha256": hashlib.sha256(
            (BASELINE / "SHA256.json").read_bytes()
        ).hexdigest(),
        "fixture": {
            "description": "Synthetic manual landmark case; no biological truth is implied.",
            "landmark_order": ["B", "A", "C30", "C20", "C10", "C5", "D"],
            "landmarks_pixels": point_map({
                name: getattr(landmarks, name)
                for name in ("A", "B", "C30", "C20", "C10", "C5", "D")
            }),
        },
        "protected_source_hashes": source_hashes,
        "measurement_row": measurement,
        "manual_bezier": {
            "fixed_fields": sorted(fin2d.FIXED_BEZIER_ENDPOINT_FIELDS),
            "draggable_fields": list(fin2d.DRAGGABLE_BEZIER_CONTROL_FIELDS),
            "controls": point_map(controls.__dict__),
            "outline_points_measurement": len(measured_outline),
            "outline_points_display": len(display_outline),
            "outline_sha256": array_sha256(measured_outline),
            "standardized_outline_sha256": array_sha256(standardized),
            "standardized_first": list(map(float, standardized[0])),
            "standardized_last": list(map(float, standardized[-1])),
        },
        "reconstruction": {
            "warnings": warnings,
            "landmarks": point_map(landmarks_3d),
            "controls": point_map(controls_3d),
            "curve_shapes": {name: list(values.shape) for name, values in curves_3d.items()},
            "curve_hashes": {name: array_sha256(values) for name, values in curves_3d.items()},
            "area_units2": float(fin3d.fin_area_S(curves_3d["outline"], landmarks_3d["B"])),
            "reference_y_values": [float(value) for value in y_values],
            "reference_y_sha256": array_sha256(y_values),
            "edge_reference_count": int(len(edge_refs)),
            "edge_reference_sha256": array_sha256(edge_refs),
        },
        "output_policy": {
            "image_precedes_csv": True,
            "example_image": "/study/manual_fixture.png",
            "example_csv": "/other/session.csv",
            "expected_directory_name": "manual_fixture_FinShape2D3D_outputs_v1.1.3",
        },
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build_fixture(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
