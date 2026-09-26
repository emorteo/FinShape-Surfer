#!/usr/bin/env python3
"""Opt-in real-Tk use case for the FinShape Surfer 1.2.2 3D window."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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


gui = load("surfer_1_2_2_live", CODE / "FinShape_Surfer_GUI_1.2.2.py")


def descendants(widget):
    result = []
    for child in widget.winfo_children():
        result.append(child)
        result.extend(descendants(child))
    return result


@unittest.skipUnless(os.environ.get("FINSHAPE_LIVE_TK") == "1", "real Tk test is opt-in")
class LiveTkHQFixUseCase(unittest.TestCase):
    def test_modes_retain_view_side_mapping_and_visible_save(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image_path = directory / "live_hq_122.png"
            yy, xx = np.indices((260, 260))
            pixels = np.empty((260, 260, 4), dtype=np.uint8)
            pixels[..., 0] = xx
            pixels[..., 1] = yy
            pixels[..., 2] = ((xx // 3 % 2) * 180 + (yy // 3 % 2) * 60).astype(np.uint8)
            pixels[..., 3] = 255
            Image.fromarray(pixels, "RGBA").save(image_path)

            app = gui.IntegratedFinShapeApp()
            try:
                app.withdraw()
                f = gui.fin2d
                landmarks = f.Landmarks(
                    B=(220.0, 220.0), A=(115.0, 55.0), C30=(35.0, 220.0),
                    C20=(52.0, 175.0), C10=(70.0, 130.0), C5=(90.0, 88.0), D=(162.0, 115.0),
                )
                app.image_path = image_path
                app.source_image_path = image_path
                app.source_image_dir = directory
                app.csv_path = directory / "live_hq_122_morphometry.csv"
                app.pil_image = f.Image.open(image_path).convert("RGBA")
                app.flip_x = False
                app.rotate_deg = 0.0
                app.crop_box = None
                app.landmarks = {
                    key: getattr(landmarks, key)
                    for key in ("A", "B", "C30", "C20", "C10", "C5", "D")
                }
                app.controls = f.build_initial_controls(landmarks)
                app.mode = "bezier"
                app.locked_after_save = False
                app._working_image_cache = None
                app._update_button_states()
                app.btn_save_meas.invoke()
                app.update()
                app.btn_show3d.invoke()
                app.update()

                state = app.last_3d_view_controls
                window = state["window"]
                controls = descendants(window)
                left = next(w for w in controls if isinstance(w, f.tk.Radiobutton) and w.cget("text") == "Left (-Z)")
                right = next(w for w in controls if isinstance(w, f.tk.Radiobutton) and w.cget("text") == "Right (+Z)")
                photo = next(w for w in controls if isinstance(w, f.tk.Radiobutton) and w.cget("text") == "High-resolution photo")
                wireframe = next(w for w in controls if isinstance(w, f.tk.Radiobutton) and w.cget("text") == "Wireframe")

                state["pitch"].set(-4.0)
                state["yaw"].set(-84.0)
                state["roll"].set(5.0)
                expected_view = (-4.0, -84.0, 5.0)
                left.invoke()
                self.assertEqual(
                    (state["pitch"].get(), state["yaw"].get(), state["roll"].get()),
                    expected_view,
                )
                photo.invoke()
                app.update()
                self.assertEqual(app.last_hq_scene.textured_skin, "negative_z_skin")
                self.assertGreater(app.last_hq_render_result.textured_pixels, 100_000)
                self.assertLess(app.last_hq_render_result.render_seconds, 2.0)
                self.assertEqual(state["save_button_text"].get(), "Save high-resolution photo...")
                self.assertEqual(state["save_button"].winfo_manager(), "pack")

                wireframe.invoke()
                photo.invoke()
                app.update()
                self.assertEqual(
                    (state["pitch"].get(), state["yaw"].get(), state["roll"].get()),
                    expected_view,
                )

                state["yaw"].set(84.0)
                right.invoke()
                app.update()
                self.assertEqual(app.last_hq_scene.textured_skin, "positive_z_skin")

                output = directory / "saved_hq_122.png"
                with patch.object(app, "_ask_save_file_in_source_dir", return_value=output):
                    state["save_button"].invoke()
                    app.update()
                self.assertTrue(output.is_file())
                with Image.open(output) as saved_image:
                    self.assertEqual(saved_image.size, gui.PHOTO_SAVE_SIZE)
                self.assertEqual(output.parent.resolve(), directory.resolve())

                preview = ARTIFACTS / "live_hq_122_right_front.png"
                app.last_hq_view_image.save(preview)
                self.assertGreater(preview.stat().st_size, 30_000)
            finally:
                app.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
