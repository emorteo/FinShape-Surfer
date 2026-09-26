#!/usr/bin/env python3
"""Unit and integrated tests for FinShape Surfer 1.2.1 HQ texture rendering."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
from PIL import Image


CODE = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gui = load("surfer_1_2_1_test", CODE / "FinShape_Surfer_GUI_1.2.1.py")
hq = load("texture_hq_0_1_0_test", CODE / "FinShapeTextureHQ_0.1.0.py")
texture = gui.base._load_module("texture_0_1_0_for_hq_test", CODE / "FinShapeTexture_0.1.0.py")
cfd = gui.base._load_module("cfd_0_1_0_for_hq_test", CODE / "FinShapeCFD_0.1.0.py")


class Status:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value


def affine_fixture(photo_side="left", flip_applied=True):
    outline = np.array([[0.0, 0.0], [-1.0, 3.0], [-3.0, 8.0], [-6.0, 8.0], [-8.0, 0.0]])
    standardized = {
        "B": [0.0, 0.0], "A": [-6.0, 8.0], "C30": [-8.0, 0.0],
        "C20": [-7.0, 2.0], "C10": [-6.0, 4.0], "C5": [-5.0, 6.0], "D": [-2.0, 5.0],
    }
    affine = np.array([[10.0, 0.0], [0.0, -10.0], [100.0, 100.0]])
    image_landmarks = {key: (np.append(value, 1.0) @ affine).tolist() for key, value in standardized.items()}
    image_outline = np.column_stack([outline, np.ones(len(outline))]) @ affine
    yy, xx = np.indices((180, 180))
    pixels = np.empty((180, 180, 3), dtype=np.uint8)
    pixels[..., 0] = xx
    pixels[..., 1] = yy
    pixels[..., 2] = (xx * 17 + yy * 31) % 256
    snapshot = {
        "working_image": Image.fromarray(pixels, "RGB"),
        "curves": {"outline": outline},
        "standardized_landmarks": standardized,
        "image_landmarks": image_landmarks,
        "image_outline": image_outline,
        "flip_applied": flip_applied,
        "fin_id": "hq_synthetic",
        "source_image": "/synthetic/source.png",
        "source_image_sha256": "0" * 64,
        "crop_box": None,
        "rotate_deg": 0.0,
    }
    base_scene = texture.build_texture_scene(snapshot, cfd, photo_side, subdivisions=0)
    return snapshot, base_scene


class HQTextureUnitTests(unittest.TestCase):
    def test_vertex_uv_is_continuous_exact_and_immutable(self):
        snapshot, base_scene = affine_fixture()
        vertices_before = base_scene.vertices.copy()
        faces_before = base_scene.faces.copy()
        scene = hq.build_hq_scene(base_scene, snapshot, texture)
        expected = texture.map_planform_to_image(base_scene.vertices[:, :2], base_scene.affine_planform_to_image)
        np.testing.assert_allclose(scene.vertex_uv_px, expected, atol=1e-12)
        np.testing.assert_array_equal(base_scene.vertices, vertices_before)
        np.testing.assert_array_equal(base_scene.faces, faces_before)
        self.assertFalse(scene.vertices.flags.writeable)
        self.assertFalse(scene.vertex_uv_px.flags.writeable)
        self.assertFalse(scene.texture_rgb.flags.writeable)

    def test_per_pixel_texture_detail_exceeds_triangle_color_limit(self):
        snapshot, base_scene = affine_fixture()
        scene = hq.build_hq_scene(base_scene, snapshot, texture)
        result = hq.render_hq_frame(scene, 800, 600, pitch=0, yaw=0, roll=0)
        pixels = np.asarray(result.image)
        background = np.all(pixels == [248, 249, 250], axis=2)
        visible_colors = np.unique(pixels[~background], axis=0)
        selected_triangles = int(np.count_nonzero(scene.face_labels == scene.textured_skin))
        self.assertGreater(result.textured_pixels, 100_000)
        self.assertGreater(len(visible_colors), selected_triangles * 3)

    def test_depth_buffer_shows_only_camera_facing_anatomical_skin(self):
        snapshot, left_base = affine_fixture("left")
        left = hq.build_hq_scene(left_base, snapshot, texture)
        left_front = hq.render_hq_frame(left, 600, 450, pitch=0, yaw=0, roll=0)
        left_back = hq.render_hq_frame(left, 600, 450, pitch=0, yaw=180, roll=0)
        self.assertGreater(left_front.textured_pixels, 80_000)
        self.assertEqual(left_back.textured_pixels, 0)

        snapshot, right_base = affine_fixture("right")
        right = hq.build_hq_scene(right_base, snapshot, texture)
        right_front = hq.render_hq_frame(right, 600, 450, pitch=0, yaw=180, roll=0)
        self.assertGreater(right_front.textured_pixels, 80_000)

    def test_rotation_is_deterministic_and_changes_projection(self):
        snapshot, base_scene = affine_fixture()
        scene = hq.build_hq_scene(base_scene, snapshot, texture)
        first = hq.render_hq_frame(scene, 500, 400, pitch=17, yaw=25, roll=-11)
        repeat = hq.render_hq_frame(scene, 500, 400, pitch=17, yaw=25, roll=-11)
        changed = hq.render_hq_frame(scene, 500, 400, pitch=-8, yaw=50, roll=7)
        self.assertEqual(first.image.tobytes(), repeat.image.tobytes())
        self.assertNotEqual(first.image.tobytes(), changed.image.tobytes())

    def test_interactive_resolution_meets_m1_latency_budget(self):
        snapshot, base_scene = affine_fixture()
        scene = hq.build_hq_scene(base_scene, snapshot, texture)
        started = time.perf_counter()
        result = hq.render_hq_frame(scene, 1000, 700, pitch=8, yaw=25, roll=0)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 2.0)
        self.assertLess(result.render_seconds, 2.0)
        self.assertEqual(result.image.size, (1000, 700))

    def test_supersampled_save_resolution_and_safety_limit(self):
        snapshot, base_scene = affine_fixture()
        scene = hq.build_hq_scene(base_scene, snapshot, texture)
        result = hq.render_hq_frame(scene, 640, 480, pitch=8, yaw=20, roll=0, supersample=2)
        self.assertEqual(result.image.size, (640, 480))
        self.assertEqual(result.supersample, 2)
        with self.assertRaisesRegex(ValueError, "16-megapixel"):
            hq.render_hq_frame(scene, 3000, 2000, pitch=0, yaw=0, roll=0, supersample=2)

    def test_metadata_reports_resolution_without_claiming_validation(self):
        snapshot, base_scene = affine_fixture()
        scene = hq.build_hq_scene(base_scene, snapshot, texture)
        result = hq.render_hq_frame(scene, 400, 300, pitch=0, yaw=0, roll=0)
        metadata = hq.renderer_metadata(scene, result)
        self.assertEqual(metadata["mesh_triangles_unchanged"], len(base_scene.faces))
        self.assertEqual(metadata["texture_interpolation"], "per-output-pixel bilinear RGB")
        self.assertIn("does not add camera calibration", metadata["hq_limitations"])

    def test_invalid_scene_and_render_parameters_are_rejected(self):
        snapshot, base_scene = affine_fixture()
        scene = hq.build_hq_scene(base_scene, snapshot, texture)
        with self.assertRaisesRegex(ValueError, "at least 64"):
            hq.render_hq_frame(scene, 20, 20, pitch=0, yaw=0, roll=0)
        with self.assertRaisesRegex(ValueError, "Supersampling"):
            hq.render_hq_frame(scene, 100, 100, pitch=0, yaw=0, roll=0, supersample=4)
        with self.assertRaisesRegex(ValueError, "finite"):
            hq.project_vertices(scene.vertices, 100, 100, pitch=float("nan"), yaw=0, roll=0)


class Surfer121IntegrationTests(unittest.TestCase):
    def test_version_extension_does_not_override_measurement_or_scientific_core(self):
        self.assertEqual(gui.INTEGRATED_VERSION, "1.2.1")
        for method in (
            "load_image", "toggle_flip", "on_rotate_change", "apply_crop", "reset_crop",
            "start_landmarks", "reset_landmarks", "on_mouse_down", "on_mouse_move", "save_measurement",
            "generate_integrated_outputs", "export_surface_dialog",
        ):
            self.assertNotIn(method, gui.IntegratedFinShapeApp.__dict__)
        self.assertTrue(issubclass(gui.IntegratedFinShapeApp, gui.base.IntegratedFinShapeApp))

    def test_prior_sources_are_unchanged(self):
        expected = {
            "FinShape_Surfer_GUI_1.2.0.py": "2241d70328b6b7f27cbbe5523fe47a144d65ce99011bc66c09228c9f63332849",
            "FinShapeTexture_0.1.0.py": "331ec39707e6a4cabc6986cdbfd632685fdce4b666d9afee3c8e2296c31e8a6f",
            "FinShape_Surfer_GUI_1.1.3.py": "8c50ef09e71dc95be887f9dbcdaeca59905990aa774dbb8d44beeae45db01f79",
            "FinShape2D_R-GUI_2.0.20.py": "5c78bd384ca1b37ada3e6eacb90664a6fdfa26e88055b76d77d333dc2d7ed7a4",
            "FinShape3D_1.1.20.py": "b108e7794f4ba8a190797ab549a1d820f8a5f934c0258b47deaac8c463c8aed2",
            "FinShapeCFD_0.1.0.py": "0b707430fc9a4ee03e3f51bf5e7588e1462ce75acd5f828cb130156beedb518d",
        }
        for filename, digest in expected.items():
            self.assertEqual(hashlib.sha256((CODE / filename).read_bytes()).hexdigest(), digest)

    def test_integrated_output_and_hq_metadata_use_image_output_directory(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            app = self._fitted_app(directory)
            row = {
                "image_name": app.image_path.name,
                "C5B_over_AB": "0.75", "C10B_over_AB": "0.70",
                "C20B_over_AB": "0.66", "C30B_over_AB": "0.72",
            }
            app.measurements = [row]
            outputs = app.generate_integrated_outputs(row)
            self.assertTrue(all(path.exists() for path in outputs))
            self.assertIn("v1.2.1", str(outputs[0].parent))

            base_scene = texture.build_texture_scene(
                app.last_integrated_texture_snapshot, cfd, "left", subdivisions=0,
            )
            scene = hq.build_hq_scene(base_scene, app.last_integrated_texture_snapshot, texture)
            result = hq.render_hq_frame(scene, 640, 480, pitch=8, yaw=0, roll=0)
            metadata_path = app._write_hq_metadata(
                hq, scene, result, texture, app.last_integrated_texture_snapshot, pitch=8, yaw=0, roll=0,
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["surfer_version"], "1.2.1")
            self.assertEqual(metadata["render_output_size_px"], [640, 480])
            self.assertEqual(metadata["render_view_degrees"], {"pitch": 8.0, "yaw": 0.0, "roll": 0.0})
            self.assertEqual(metadata_path.parent.resolve(), outputs[0].parent.resolve())
            self.assertIs(metadata["camera_pose_estimated"], False)

            image_path = ARTIFACTS / "hq_software_texture_left.png"
            result.image.save(image_path)
            self.assertGreater(image_path.stat().st_size, 20_000)

    @staticmethod
    def _fitted_app(directory: Path):
        f = gui.fin2d
        app = object.__new__(gui.IntegratedFinShapeApp)
        landmarks = f.Landmarks(
            B=(180.0, 180.0), A=(100.0, 60.0), C30=(40.0, 180.0),
            C20=(52.0, 150.0), C10=(65.0, 120.0), C5=(78.0, 90.0), D=(135.0, 105.0),
        )
        app.landmarks = {key: getattr(landmarks, key) for key in ("A", "B", "C30", "C20", "C10", "C5", "D")}
        app.controls = f.build_initial_controls(landmarks)
        yy, xx = np.indices((220, 220))
        pixels = np.empty((220, 220, 4), dtype=np.uint8)
        pixels[..., 0] = xx
        pixels[..., 1] = yy
        pixels[..., 2] = (xx * 17 + yy * 31) % 256
        pixels[..., 3] = 255
        app.pil_image = f.Image.fromarray(pixels, "RGBA")
        app.flip_x = True
        app.rotate_deg = 0.0
        app.crop_box = None
        app._working_image_cache = None
        app._fin3d_module = None
        app._texture_module = None
        app._hq_texture_module = None
        app.last_integrated_3d_path = None
        app.last_integrated_curves2d = None
        app.last_integrated_fin_id = None
        app.last_integrated_surface_snapshot = None
        app.last_integrated_texture_snapshot = None
        app.last_texture_metadata_path = None
        app.last_hq_scene = None
        app.last_hq_render_result = None
        app.last_hq_view_image = None
        app.last_hq_metadata_path = None
        app.btn_show3d = None
        app.status = Status()
        directory.mkdir(parents=True, exist_ok=True)
        app.image_path = directory / "hq_synthetic.png"
        app.pil_image.save(app.image_path)
        app.source_image_path = app.image_path
        app.source_image_dir = directory
        app.csv_path = directory / "hq_synthetic_morphometry.csv"
        app.measurements = []
        return app


if __name__ == "__main__":
    unittest.main(verbosity=2)
