#!/usr/bin/env python3
"""Unit and regression tests for FinShape Surfer 1.2.2 display fixes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
import unittest

import numpy as np


CODE = Path(__file__).resolve().parents[1]
PRIOR_TEST = CODE / "Surfer_validation_1.2.1" / "test_hq_texture_renderer.py"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prior = load("surfer_121_fixture_for_122", PRIOR_TEST)
gui = load("surfer_1_2_2_test", CODE / "FinShape_Surfer_GUI_1.2.2.py")
hq = load("texture_hq_0_2_0_test", CODE / "FinShapeTextureHQ_0.2.0.py")
texture = gui.base.base._load_module("texture_0_1_0_for_122_test", CODE / "FinShapeTexture_0.1.0.py")
cfd = gui.base.base._load_module("cfd_0_1_0_for_122_test", CODE / "FinShapeCFD_0.1.0.py")


class HQViewFixTests(unittest.TestCase):
    def test_left_and_right_map_to_corrected_geometric_skins(self):
        snapshot, left_base = prior.affine_fixture("left")
        left = hq.build_hq_scene(left_base, snapshot, texture)
        snapshot, right_base = prior.affine_fixture("right")
        right = hq.build_hq_scene(right_base, snapshot, texture)
        self.assertEqual(left.textured_skin, "negative_z_skin")
        self.assertEqual(right.textured_skin, "positive_z_skin")

        left_front = hq.render_hq_frame(left, 600, 450, pitch=0, yaw=-90, roll=0, supersample=2)
        right_front = hq.render_hq_frame(right, 600, 450, pitch=0, yaw=90, roll=0, supersample=2)
        self.assertGreater(left_front.textured_pixels, 200_000)
        self.assertGreater(right_front.textured_pixels, 200_000)

    def test_edge_wall_faces_are_not_rendered_in_photo_view(self):
        snapshot, base_scene = prior.affine_fixture("left")
        scene = hq.build_hq_scene(base_scene, snapshot, texture)
        edge_ids = np.flatnonzero(scene.face_labels == "edge_wall")
        self.assertGreater(len(edge_ids), 0)
        edge_only = replace(
            scene,
            faces=scene.faces[edge_ids],
            face_labels=scene.face_labels[edge_ids],
        )
        result = hq.render_hq_frame(edge_only, 400, 300, pitch=10, yaw=80, roll=0)
        expected = np.empty((300, 400, 3), dtype=np.uint8)
        expected[:] = [248, 249, 250]
        np.testing.assert_array_equal(np.asarray(result.image), expected)

    def test_two_times_supersampling_adds_silhouette_antialiasing(self):
        snapshot, base_scene = prior.affine_fixture("right")
        scene = hq.build_hq_scene(base_scene, snapshot, texture)
        flat_texture = np.empty_like(scene.texture_rgb)
        flat_texture[:] = [210, 20, 40]
        flat_texture.setflags(write=False)
        scene = replace(scene, texture_rgb=flat_texture)
        plain = hq.render_hq_frame(scene, 520, 390, pitch=-4, yaw=-84, roll=5, supersample=1)
        smooth = hq.render_hq_frame(scene, 520, 390, pitch=-4, yaw=-84, roll=5, supersample=2)
        plain_colors = len(np.unique(np.asarray(plain.image).reshape(-1, 3), axis=0))
        smooth_colors = len(np.unique(np.asarray(smooth.image).reshape(-1, 3), axis=0))
        self.assertGreater(smooth_colors, plain_colors)

        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        smooth.image.save(ARTIFACTS / "antialiased_oblique_silhouette.png")

    def test_antialiased_interactive_scene_stays_within_m1_budget(self):
        snapshot, _ = prior.affine_fixture("right")
        display_base = texture.build_texture_scene(snapshot, cfd, "right", subdivisions=0)
        scene = hq.build_hq_scene(display_base, snapshot, texture)
        started = time.perf_counter()
        result = hq.render_hq_frame(scene, 1000, 700, pitch=8, yaw=25, roll=0, supersample=2)
        elapsed = time.perf_counter() - started
        self.assertEqual(len(scene.faces), len(display_base.faces))
        self.assertLess(elapsed, 3.0)
        self.assertEqual(result.image.size, (1000, 700))

    def test_camera_yaw_matches_matplotlib_azimuth_convention(self):
        snapshot, base_scene = prior.affine_fixture("left")
        scene = hq.build_hq_scene(base_scene, snapshot, texture)
        broad, _ = hq.project_vertices(scene.vertices, 800, 600, pitch=0, yaw=-90, roll=0)
        edge, _ = hq.project_vertices(scene.vertices, 800, 600, pitch=0, yaw=0, roll=0)
        self.assertGreater(np.ptp(broad[:, 0]), np.ptp(edge[:, 0]) * 4.0)

    def test_metadata_records_display_only_changes(self):
        snapshot, base_scene = prior.affine_fixture("left")
        scene = hq.build_hq_scene(base_scene, snapshot, texture)
        result = hq.render_hq_frame(scene, 400, 300, pitch=0, yaw=180, roll=0, supersample=2)
        metadata = hq.renderer_metadata(scene, result)
        self.assertEqual(metadata["geometric_skin"], "negative_z_skin")
        self.assertIs(metadata["edge_wall_rendered_in_photo_view"], False)
        self.assertIs(metadata["back_face_culling"], True)
        self.assertIs(metadata["cfd_surface_changed"], False)

    def test_new_gui_changes_no_measurement_or_cfd_method(self):
        self.assertEqual(gui.INTEGRATED_VERSION, "1.2.2")
        for method in (
            "load_image", "toggle_flip", "on_rotate_change", "apply_crop", "reset_crop",
            "start_landmarks", "reset_landmarks", "on_mouse_down", "on_mouse_move",
            "save_measurement", "generate_integrated_outputs", "export_surface_dialog",
        ):
            self.assertNotIn(method, gui.IntegratedFinShapeApp.__dict__)
        self.assertTrue(issubclass(gui.IntegratedFinShapeApp, gui.base.IntegratedFinShapeApp))

    def test_protected_sources_are_unchanged(self):
        expected = {
            "FinShape_Surfer_GUI_1.2.1.py": "4cf04584d19ec118dec6ce58362a6f714fc95531a9a697919829ee2dc01f3322",
            "FinShapeTextureHQ_0.1.0.py": "ce2ca4ff98180cf891abbb29d9f88b04db3189ff6732072f92a2598509bfa917",
            "FinShape_Surfer_GUI_1.2.0.py": "2241d70328b6b7f27cbbe5523fe47a144d65ce99011bc66c09228c9f63332849",
            "FinShapeTexture_0.1.0.py": "331ec39707e6a4cabc6986cdbfd632685fdce4b666d9afee3c8e2296c31e8a6f",
            "FinShape_Surfer_GUI_1.1.3.py": "8c50ef09e71dc95be887f9dbcdaeca59905990aa774dbb8d44beeae45db01f79",
            "FinShape2D_R-GUI_2.0.20.py": "5c78bd384ca1b37ada3e6eacb90664a6fdfa26e88055b76d77d333dc2d7ed7a4",
            "FinShape3D_1.1.20.py": "b108e7794f4ba8a190797ab549a1d820f8a5f934c0258b47deaac8c463c8aed2",
            "FinShapeCFD_0.1.0.py": "0b707430fc9a4ee03e3f51bf5e7588e1462ce75acd5f828cb130156beedb518d",
        }
        for filename, expected_hash in expected.items():
            self.assertEqual(hashlib.sha256((CODE / filename).read_bytes()).hexdigest(), expected_hash)


if __name__ == "__main__":
    unittest.main(verbosity=2)
