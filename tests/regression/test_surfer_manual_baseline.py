#!/usr/bin/env python3
"""Golden regression tests for the reviewed FinShape Surfer 1.2.2 workflow."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np


os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "baselines" / "FinShape_Surfer_1.2.2"
CODE = BASELINE / "Code"
GOLDEN_PATH = ROOT / "tests" / "fixtures" / "surfer_1_2_2" / "manual_workflow_golden.json"


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


fin2d = load_module("finshape_2d_2_0_20_regression", CODE / "FinShape2D_R-GUI_2.0.20.py")
fin3d = load_module("finshape_3d_1_1_20_regression", CODE / "FinShape3D_1.1.20.py")
integrated = load_module("finshape_surfer_1_1_3_regression", CODE / "FinShape_Surfer_GUI_1.1.3.py")
golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def fixture_landmarks():
    points = golden["fixture"]["landmarks_pixels"]
    return fin2d.Landmarks(**{name: tuple(points[name]) for name in points})


def measurement_row(landmarks, controls):
    ratios = fin2d.compute_weller_ratios(landmarks)
    outline = fin2d.outline_polygon(controls, n_per_seg=900)
    row = {
        "image_name": "manual_fixture.png",
        "AB_px": f"{ratios['AB_px']:.6f}",
        "area_px2": f"{fin2d.shoelace_area(outline):.6f}",
        "C5B_over_AB": f"{ratios['C5B_over_AB']:.10f}",
        "C10B_over_AB": f"{ratios['C10B_over_AB']:.10f}",
        "C20B_over_AB": f"{ratios['C20B_over_AB']:.10f}",
        "C30B_over_AB": f"{ratios['C30B_over_AB']:.10f}",
        "DB_over_AB": f"{ratios['DB_over_AB']:.10f}",
    }
    for name in ("A", "C5", "C10", "C20", "C30", "D"):
        point = getattr(landmarks, name)
        x, y = point[0] - landmarks.B[0], point[1] - landmarks.B[1]
        row[f"{name}_dx"] = f"{x:.6f}"
        row[f"{name}_dy"] = f"{y:.6f}"
    return row


class SurferManualBaselineTests(unittest.TestCase):
    maxDiff = None

    def test_release_files_match_frozen_sha256_manifest(self):
        expected = golden["protected_source_hashes"]
        self.assertTrue(expected)
        for relative, digest in expected.items():
            path = BASELINE / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, relative)

    def test_fixed_and_draggable_bezier_roles_are_frozen(self):
        self.assertEqual(
            sorted(fin2d.FIXED_BEZIER_ENDPOINT_FIELDS),
            golden["manual_bezier"]["fixed_fields"],
        )
        self.assertEqual(
            list(fin2d.DRAGGABLE_BEZIER_CONTROL_FIELDS),
            golden["manual_bezier"]["draggable_fields"],
        )
        self.assertNotIn("LE0_Q2", fin2d.DRAGGABLE_BEZIER_CONTROL_FIELDS)

    def test_landmarks_controls_measurement_and_area_match_golden(self):
        landmarks = fixture_landmarks()
        controls = fin2d.build_initial_controls(landmarks)
        self.assertEqual(measurement_row(landmarks, controls), golden["measurement_row"])
        for name, expected in golden["manual_bezier"]["controls"].items():
            np.testing.assert_allclose(getattr(controls, name), expected, rtol=0.0, atol=1e-12)

    def test_measured_and_standardized_contours_match_golden(self):
        landmarks = fixture_landmarks()
        controls = fin2d.build_initial_controls(landmarks)
        measured = fin2d.outline_polygon(controls, n_per_seg=900)
        displayed = fin2d.outline_polygon(controls, n_per_seg=1100)
        standardized = fin2d.standardize_contour_units(
            displayed, landmarks.B, landmarks.A, landmarks.C30, ab_units=10.0
        )
        expected = golden["manual_bezier"]
        self.assertEqual(len(measured), expected["outline_points_measurement"])
        self.assertEqual(len(displayed), expected["outline_points_display"])
        self.assertEqual(array_sha256(measured), expected["outline_sha256"])
        self.assertEqual(array_sha256(standardized), expected["standardized_outline_sha256"])
        np.testing.assert_allclose(standardized[0], expected["standardized_first"], atol=1e-12)
        np.testing.assert_allclose(standardized[-1], expected["standardized_last"], atol=1e-12)
        ab = np.linalg.norm(np.asarray(landmarks.A) - np.asarray(landmarks.B))
        self.assertGreater(ab, 0.0)

    def test_ratio_driven_2d_and_wireframe_references_match_golden(self):
        landmarks = fixture_landmarks()
        controls = fin2d.build_initial_controls(landmarks)
        row = measurement_row(landmarks, controls)
        series = integrated.measurement_row_to_3d_series(row, fin3d, "manual_fixture")
        warnings = []
        lm3d, u_ab = fin3d.build_landmarks(series, BC30_scale=10.0)
        ctrl3d = fin3d.build_controls(lm3d, u_ab, warnings=warnings)
        curves = fin3d.compute_2d_curves(ctrl3d, n=900)
        expected = golden["reconstruction"]
        self.assertEqual(warnings, expected["warnings"])
        for name, values in expected["landmarks"].items():
            np.testing.assert_allclose(lm3d[name], values, rtol=0.0, atol=1e-12)
        for name, values in expected["controls"].items():
            np.testing.assert_allclose(ctrl3d[name], values, rtol=0.0, atol=1e-12)
        self.assertEqual({k: list(v.shape) for k, v in curves.items()}, expected["curve_shapes"])
        self.assertEqual({k: array_sha256(v) for k, v in curves.items()}, expected["curve_hashes"])
        self.assertAlmostEqual(
            fin3d.fin_area_S(curves["outline"], lm3d["B"]), expected["area_units2"], places=12
        )
        y_max = float(curves["outline"][:, 1].max())
        y_values = fin3d.contour_reference_y_values(
            curves, y_max, n_y_slices=39, top_focus_power=3.0,
            n_edge_points=100, max_sections=55,
        )
        edge_refs = fin3d.contour_reference_points(curves, n_edge_points=100)
        self.assertEqual(array_sha256(y_values), expected["reference_y_sha256"])
        self.assertEqual(len(edge_refs), expected["edge_reference_count"])
        self.assertEqual(array_sha256(edge_refs), expected["edge_reference_sha256"])
        self.assertEqual(float(y_values[0]), 0.0)
        self.assertAlmostEqual(float(y_values[-1]), y_max, places=9)

    def test_output_directory_uses_source_image_directory(self):
        policy = golden["output_policy"]
        image = Path(policy["example_image"])
        csv = Path(policy["example_csv"])
        output = integrated.output_dir_for_measurement(csv, image)
        self.assertEqual(output.parent, image.parent)
        self.assertEqual(output.name, policy["expected_directory_name"])

    def test_current_display_extension_does_not_override_measurement_methods(self):
        current = load_module(
            "finshape_surfer_1_2_2_regression",
            CODE / "FinShape_Surfer_GUI_1.2.2.py",
        )
        self.assertEqual(current.INTEGRATED_VERSION, "1.2.2")
        for method in (
            "load_image", "toggle_flip", "on_rotate_change", "apply_crop", "reset_crop",
            "start_landmarks", "reset_landmarks", "on_mouse_down", "on_mouse_move",
            "save_measurement", "generate_integrated_outputs", "export_surface_dialog",
        ):
            self.assertNotIn(method, current.IntegratedFinShapeApp.__dict__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
