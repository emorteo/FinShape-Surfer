#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinShape2D R-GUI 2.0.13

Fixes vs 2.0.7 (requested)
--------------------------
1) Button workflow order + Apply Crop activation:
   - Step order: 1 Load, 2 Flip, 3 Rotate, 4 Crop, 5 Apply Crop (enabled only after a crop box is defined), 6 Reset Crop.
2) Row labels:
   - Row 1 labeled **Edit image**
   - Row 2 labeled **Morphometry**
3) Current image name displayed at the upper-right of the main window.
4) Saving filenames:
   - When choosing the CSV file for the session, default filename is based on the CURRENT image name.
   - Contour exports default to "<image_stem>_contour.<ext>".
5) 2D display y-axis:
   - Uses a true y-up plotting convention (values increase upwards on screen) by flipping y during canvas mapping.
6) Save 2D contour (rotated) in common formats:
   - Adds "Save contour…" button in the 2D window.
   - Saves rotated so that segment B→C30 is perfectly horizontal (to the right).
   - Ensures fin "swims right" (if tip A ends up on the left, it mirrors x).
   - Supports SVG export natively (no extra deps).
   - Supports PNG/JPG/BMP exports if Pillow is installed (via PostScript rasterization; may require Ghostscript on some systems).

Dependencies
------------
- Standard library: tkinter, math, csv, pathlib, dataclasses.
- Optional: Pillow (PIL) for JPEG + flip/rotate/crop + zoom + raster exports.
  Install: pip install pillow
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PIL_OK = False
try:
    from PIL import Image, ImageTk, ImageDraw  # type: ignore
    PIL_OK = True
except Exception:
    PIL_OK = False


from core.math.geometry import clamp, v_add, v_sub, v_mul, v_len, v_norm, dist, shoelace_area, rot2, rotate_about_origin
from core.math.bezier import bezier_quad, bezier_cubic, sample_curve

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import subprocess
import platform
import pandas as pd
import shutil
import webbrowser

from core.reconstruction.wireframe import build_landmarks, build_controls, compute_2d_curves
from core.ui.plotting_3d import plot_2d, plot_3d_wireframe


"""Graphical user interface for FinShape2D morphometry.

The `FinShape2DApp` class provides an interactive Tk-based workflow for
landmarking a dorsal fin image, fitting Bézier controls, saving morphometric
ratios to CSV, and invoking the FinShape3D reconstruction pipeline.

Design notes for contributors:
- Landmarks are collected in the order: B → A → C30 → C20 → C10 → C5 → D
- `save_measurement()` writes a CSV row compatible with FinShape3D's
    `build_landmarks` by including either r_* ratios or C*B_over_AB variants.
- `generate_3d()` builds a pandas Series and calls the reconstruction API.
"""


























@dataclass
class Landmarks:
    A: Tuple[float, float]
    B: Tuple[float, float]
    C5: Tuple[float, float]
    C10: Tuple[float, float]
    C20: Tuple[float, float]
    C30: Tuple[float, float]
    D: Tuple[float, float]


@dataclass
class BezierControls:
    LE0_Q0: Tuple[float, float]
    LE0_Q1: Tuple[float, float]
    LE0_Q2: Tuple[float, float]
    LE1_P0: Tuple[float, float]
    LE1_P1: Tuple[float, float]
    LE1_P2: Tuple[float, float]
    LE1_P3: Tuple[float, float]
    TE1_P0: Tuple[float, float]
    TE1_P1: Tuple[float, float]
    TE1_P2: Tuple[float, float]
    TE1_P3: Tuple[float, float]
    TE2_Q0: Tuple[float, float]
    TE2_Q1: Tuple[float, float]
    TE2_Q2: Tuple[float, float]
    TE3_Q0: Tuple[float, float]
    TE3_Q1: Tuple[float, float]
    TE3_Q2: Tuple[float, float]


def build_initial_controls(lm: Landmarks) -> BezierControls:
    """Generate default Bézier controls from user-placed landmarks.

    This provides a reasonable initial fit so users can fine-tune the
    control points interactively. The function returns a `BezierControls`
    dataclass with named control points for the leading and trailing edges.
    """

    A, B, D = lm.A, lm.B, lm.D
    C5, C10, C20, C30 = lm.C5, lm.C10, lm.C20, lm.C30

    uAB = v_norm(v_sub(A, B))
    nAB = rot2(uAB, -math.pi / 2.0)

    BD = dist(B, D)
    LE0_Q0, LE0_Q2 = B, D
    LE0_Q1 = v_add(B, v_add(v_mul(v_norm(v_sub(D, B)), 0.35 * BD), v_mul(nAB, -0.08 * BD)))

    DA = dist(D, A)
    uDA = v_norm(v_sub(A, D))
    nDA = rot2(uDA, -math.pi / 2.0)
    LE1_P0, LE1_P3 = D, A
    LE1_P1 = v_add(D, v_add(v_mul(uDA, 0.25 * DA), v_mul(nDA, -0.10 * DA)))
    LE1_P2 = v_add(A, v_add(v_mul(uDA, -0.20 * DA), v_mul(nDA, -0.07 * DA)))

    AC5 = dist(A, C5)
    uAC5 = v_norm(v_sub(C5, A))
    nAC5 = rot2(uAC5, math.pi / 2.0)
    TE1_P1 = v_add(A, v_add(v_mul(uAC5, 0.35 * AC5), v_mul(nAC5, 0.15 * AC5)))
    TE1_P2 = v_add(C5, v_add(v_mul(uAC5, -0.25 * AC5), v_mul(nAC5, 0.08 * AC5)))

    TE2_Q1 = (2 * C10[0] - 0.5 * (C5[0] + C20[0]), 2 * C10[1] - 0.5 * (C5[1] + C20[1]))

    mid = ((C20[0] + C30[0]) / 2.0, (C20[1] + C30[1]) / 2.0)
    u = v_norm(v_sub(C30, C20))
    n = rot2(u, math.pi / 2.0)
    seg = dist(C20, C30)
    TE3_Q1 = v_add(mid, v_mul(n, 0.08 * seg))

    return BezierControls(
        LE0_Q0=LE0_Q0, LE0_Q1=LE0_Q1, LE0_Q2=LE0_Q2,
        LE1_P0=D, LE1_P1=LE1_P1, LE1_P2=LE1_P2, LE1_P3=A,
        TE1_P0=A, TE1_P1=TE1_P1, TE1_P2=TE1_P2, TE1_P3=C5,
        TE2_Q0=C5, TE2_Q1=TE2_Q1, TE2_Q2=C20,
        TE3_Q0=C20, TE3_Q1=TE3_Q1, TE3_Q2=C30
    )


def curves_from_controls(ctrl: BezierControls, n_per_seg: int = 320) -> Dict[str, List[Tuple[float, float]]]:
    """Sample the tuple-based Bézier control points into polyline segments.

    Returns a dict mapping segment names to lists of (x,y) tuples usable for
    drawing on the Tkinter canvas.
    """
    le0 = sample_curve(lambda t: bezier_quad(ctrl.LE0_Q0, ctrl.LE0_Q1, ctrl.LE0_Q2, t), n_per_seg)
    le1 = sample_curve(lambda t: bezier_cubic(ctrl.LE1_P0, ctrl.LE1_P1, ctrl.LE1_P2, ctrl.LE1_P3, t), n_per_seg)
    te1 = sample_curve(lambda t: bezier_cubic(ctrl.TE1_P0, ctrl.TE1_P1, ctrl.TE1_P2, ctrl.TE1_P3, t), n_per_seg)
    te2 = sample_curve(lambda t: bezier_quad(ctrl.TE2_Q0, ctrl.TE2_Q1, ctrl.TE2_Q2, t), n_per_seg)
    te3 = sample_curve(lambda t: bezier_quad(ctrl.TE3_Q0, ctrl.TE3_Q1, ctrl.TE3_Q2, t), n_per_seg)
    return {"LE0": le0, "LE1": le1, "TE1": te1, "TE2": te2, "TE3": te3}


def outline_polygon(ctrl: BezierControls, n_per_seg: int = 500) -> List[Tuple[float, float]]:
    """Return a continuous polygon outlining the fin contour.

    The ordering stitches leading and trailing edge segments and removes
    duplicated junction points so the returned list is suitable for area
    computations and continuous drawing.
    """
    c = curves_from_controls(ctrl, n_per_seg)
    le0 = c["LE0"]
    le1 = c["LE1"][1:]
    te = (c["TE1"][:-1] + c["TE2"][:-1] + c["TE3"])[1:]
    return le0 + le1 + te


def compute_weller_ratios(lm: Landmarks) -> Dict[str, float]:
    AB = dist(lm.A, lm.B)
    if AB < 1e-9:
        raise ValueError("AB length is zero.")
    return {
        "AB_px": AB,
        "C5B_over_AB": dist(lm.C5, lm.B) / AB,
        "C10B_over_AB": dist(lm.C10, lm.B) / AB,
        "C20B_over_AB": dist(lm.C20, lm.B) / AB,
        "C30B_over_AB": dist(lm.C30, lm.B) / AB,
        "DB_over_AB": dist(lm.D, lm.B) / AB,
    }


def clip_ray_to_rect(origin: Tuple[float, float], direction: Tuple[float, float], rect: Tuple[float, float, float, float]):
    ox, oy = origin
    dx, dy = direction
    xmin, ymin, xmax, ymax = rect
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return None
    t0, t1 = 0.0, float("inf")

    def clip(p, q):
        nonlocal t0, t1
        if abs(p) < 1e-12:
            return q >= 0
        r = q / p
        if p < 0:
            if r > t1:
                return False
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return False
            if r < t1:
                t1 = r
        return True

    if not clip(-dx, ox - xmin): return None
    if not clip(dx, xmax - ox):  return None
    if not clip(-dy, oy - ymin): return None
    if not clip(dy, ymax - oy):  return None
    if t1 < t0: return None
    p0 = (ox + t0*dx, oy + t0*dy)
    if math.isinf(t1): t1 = t0 + 1e6
    p1 = (ox + t1*dx, oy + t1*dy)
    return p0, p1




def rotate_and_orient_swim_right(
    pts_rel_px: List[Tuple[float, float]],
    A_rel_px: Tuple[float, float],
    C30_rel_px: Tuple[float, float],
) -> Tuple[List[Tuple[float, float]], Tuple[float, float], Tuple[float, float]]:
    ang = -math.atan2(C30_rel_px[1], C30_rel_px[0])
    pts_r = [rotate_about_origin(p, ang) for p in pts_rel_px]
    A_r = rotate_about_origin(A_rel_px, ang)
    C30_r = rotate_about_origin(C30_rel_px, ang)
    if A_r[0] < 0:
        pts_r = [(-p[0], p[1]) for p in pts_r]
        A_r = (-A_r[0], A_r[1])
        C30_r = (-C30_r[0], C30_r[1])
    return pts_r, A_r, C30_r


class FinShape2DApp(tk.Tk):
    CSV_COLUMNS = [
        "id",
        "r_BA",
        "r_BC5",
        "r_BC10",
        "r_BC20",
        "r_BC30",
        "AB_px",
        "area_px2",
        "DB_over_AB",
        "A_dx", "A_dy",
        "C5_dx", "C5_dy",
        "C10_dx", "C10_dy",
        "C20_dx", "C20_dy",
        "C30_dx", "C30_dy",
        "D_dx", "D_dy",
        "image_name",
    ]

    def __init__(self):
        super().__init__()
        self.title("FinShape2D R-GUI 2.0.13")
        self.geometry("1280x820")

        self.image_path: Optional[Path] = None
        self.pil_image = None
        self.tk_image: Optional[tk.PhotoImage] = None
        self.display_scale = 1.0

        self.crop_box: Optional[Tuple[int, int, int, int]] = None
        self.flip_x = False
        self.rotate_deg = 0.0

        self.landmarks: Dict[str, Tuple[float, float]] = {}
        self.controls: Optional[BezierControls] = None
        self.mode = "idle"
        self._drag_start = None
        self._crop_rect_id = None
        self._active_handle = None

        self.csv_path: Optional[Path] = None
        self.measurements: List[Dict[str, str]] = []

        self.zoom_half = 22
        self.zoom_w, self.zoom_h = 240, 200
        self.zoom_imgtk = None


        self.locked_after_save: bool = False
        self.btn_apply_crop: Optional[tk.Button] = None
        self.lbl_image_name: Optional[tk.Label] = None

        self.btn_flip = None
        self.scale_rotate = None
        self.btn_crop = None
        self.btn_reset_crop = None
        self.btn_set_landmarks = None
        self.btn_save_meas = None
        self.btn_new_image = None
        self.btn_exit = None
        self._build_ui()
        self._setup_zoom()

        self._update_button_states()
    def _build_ui(self):
        bar = tk.Frame(self, bd=1, relief="raised")
        bar.pack(side="top", fill="x")

        row1 = tk.Frame(bar); row1.pack(side="top", fill="x")
        row2 = tk.Frame(bar); row2.pack(side="top", fill="x")

        tk.Label(row1, text="Edit image", font=("Helvetica", 12, "bold")).pack(side="left", padx=(10, 6))
        tk.Button(row1, text="1. Load Image", command=self.load_image).pack(side="left", padx=4, pady=3)
        self.btn_flip = tk.Button(row1, text="2. Flip ↔", command=self.toggle_flip)
        self.btn_flip.pack(side="left", padx=4, pady=3)

        tk.Label(row1, text="3. Rotate").pack(side="left", padx=(10, 0))
        self.rot_var = tk.DoubleVar(value=0.0)
        self.scale_rotate = tk.Scale(row1, from_=-30, to=30, orient="horizontal", resolution=0.5,
                 variable=self.rot_var, length=170, command=self.on_rotate_change)
        self.scale_rotate.pack(side="left", padx=4)

        self.btn_crop = tk.Button(row1, text="4. Crop", command=self.enter_crop)
        self.btn_crop.pack(side="left", padx=(10, 4), pady=3)

        self.btn_apply_crop = tk.Button(row1, text="5. Apply Crop", command=self.apply_crop, state="disabled")
        self.btn_apply_crop.pack(side="left", padx=4, pady=3)

        self.btn_reset_crop = tk.Button(row1, text="6. Reset Crop", command=self.reset_crop)
        self.btn_reset_crop.pack(side="left", padx=4, pady=3)

        self.lbl_image_name = tk.Label(row1, text="", anchor="e")
        self.lbl_image_name.pack(side="right", padx=10)

        tk.Label(row2, text="Morphometry", font=("Helvetica", 12, "bold")).pack(side="left", padx=(10, 6))
        self.btn_set_landmarks = tk.Button(row2, text="7. Set Landmarks", command=self.start_landmarks)
        self.btn_set_landmarks.pack(side="left", padx=4, pady=3)
        self.btn_save_meas = tk.Button(row2, text="8. Save Measurement (CSV)", command=self.save_measurement)
        self.btn_save_meas.pack(side="left", padx=4, pady=3)
        self.btn_gen_3d = tk.Button(row2, text="8.1 Generate 3D", command=self.generate_3d, state="disabled")
        self.btn_gen_3d.pack(side="left", padx=4, pady=3)
        self.btn_show2d = tk.Button(row2, text="9. Show 2D", command=self.show_2d_window, state="disabled")
        self.btn_show2d.pack(side="left", padx=4, pady=3)
        self.btn_new_image = tk.Button(row2, text="10. Measure new image", command=self.measure_new_image)
        self.btn_new_image.pack(side="left", padx=14, pady=3)
        self.btn_exit = tk.Button(row2, text="Exit (Save CSV)", command=self.exit_save)
        self.btn_exit.pack(side="right", padx=10, pady=3)

        self.status = tk.StringVar(value="Load an image to begin.")
        tk.Label(self, textvariable=self.status, anchor="w").pack(side="bottom", fill="x")

        self.canvas = tk.Canvas(self, bg="#222", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        side = tk.Frame(self, width=360); side.pack(side="right", fill="y")
        self.info = tk.Text(side, height=34, width=58)
        self.info.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        self.info.insert("end",
                         "Instructions\n\n"
                         "• Zoom (lower-right): move mouse over image to see 5× zoom with stable rays.\n"
                         "• IMPORTANT: click using the TIP of the mouse cursor.\n\n"
                         "Edit image: Flip / Rotate / Crop.\n"
                         "  Apply Crop activates only after you draw a crop box.\n\n"
                         "Morphometry:\n"
                         "  Landmarks: B → A → C30 → C20 → C10 → C5 → D\n"
                         "  After D, Bézier control points appear automatically.\n"
                         "  Drag orange control points to fit contour.\n"
                         "  Save Measurement updates the session CSV.\n"
                         "  Show 2D displays rotated contour (B→C30 horizontal) with axes (AB=10).\n")
        self.info.configure(state="disabled")

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def _setup_zoom(self):
        self.zoom_canvas = tk.Canvas(self, width=self.zoom_w, height=self.zoom_h, bg="black", highlightthickness=1)
        self.zoom_canvas.place(relx=1.0, rely=1.0, x=-10, y=-40, anchor="se")
        self.zoom_canvas.create_text(6, 6, anchor="nw", text="5× zoom", fill="white", font=("Helvetica", 10, "bold"))

    def _disable_2d(self):
        self.btn_show2d.configure(state="disabled")

    def _enable_2d(self):
        self.btn_show2d.configure(state="normal")


    def _update_button_states(self):
        """Enable/disable controls depending on whether measurement has been saved for current image."""
        locked = getattr(self, "locked_after_save", False)

        edit_state = "disabled" if locked else "normal"
        for w in [getattr(self, "btn_flip", None), getattr(self, "btn_crop", None),
                  getattr(self, "btn_reset_crop", None), getattr(self, "scale_rotate", None)]:
            if w is not None:
                try:
                    w.configure(state=edit_state)
                except Exception:
                    pass

        if self.btn_apply_crop is not None:
            self.btn_apply_crop.configure(state=("normal" if (self.crop_box is not None and not locked) else "disabled"))

        for w in [getattr(self, "btn_set_landmarks", None), getattr(self, "btn_save_meas", None)]:
            if w is not None:
                w.configure(state=("disabled" if locked else "normal"))

        if getattr(self, "btn_gen_3d", None) is not None:
            self.btn_gen_3d.configure(state=("normal" if locked else "disabled"))

        if getattr(self, "btn_show2d", None) is not None:
            self.btn_show2d.configure(state=("normal" if locked else "disabled"))

        if getattr(self, "btn_new_image", None) is not None:
            self.btn_new_image.configure(state="normal")

        if getattr(self, "btn_exit", None) is not None:
            self.btn_exit.configure(state="normal")

    def measure_new_image(self):
        """Reset the workflow and prompt for a new image."""
        self.image_path = None
        self.pil_image = None
        self.tk_image = None
        self.canvas.delete("all")

        self.crop_box = None
        self.flip_x = False
        self.rotate_deg = 0.0
        self.rot_var.set(0.0)

        self.landmarks.clear()
        self.controls = None
        self.mode = "idle"
        self._active_handle = None
        self._crop_rect_id = None
        self._drag_start = None

        self.locked_after_save = False
        self._disable_2d()

        if self.lbl_image_name is not None:
            self.lbl_image_name.configure(text="")

        self.status.set("Step 1: Load Image.")
        self._update_button_states()
        self.load_image()

    def _default_csv_name(self) -> str:
        if self.image_path is None:
            return "morphometry.csv"
        return f"{self.image_path.stem}_morphometry.csv"

    def _ensure_csv_path(self) -> Optional[Path]:
        if self.csv_path is not None:
            return self.csv_path
        p = filedialog.asksaveasfilename(
            title="Choose CSV file for all measurements",
            initialfile=self._default_csv_name(),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not p:
            return None
        self.csv_path = Path(p)
        return self.csv_path

    def _write_csv(self):
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            w.writeheader()
            for row in self.measurements:
                w.writerow(row)

    def load_image(self):
        p = filedialog.askopenfilename(
            title="Select dorsal fin image",
            filetypes=[("Images", "*.png *.gif *.ppm *.pgm *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")]
        )
        if not p:
            return
        self.image_path = Path(p)
        self.locked_after_save = False
        if self.lbl_image_name is not None:
            self.lbl_image_name.configure(text=self.image_path.name)

        self.crop_box = None
        self.flip_x = False
        self.rotate_deg = 0.0
        self.rot_var.set(0.0)
        self.landmarks.clear()
        self.controls = None
        self.mode = "idle"
        self._active_handle = None
        self._crop_rect_id = None
        self._drag_start = None
        self._disable_2d()
        if self.btn_apply_crop is not None:
            self.btn_apply_crop.configure(state="disabled")

        try:
            self._load_image_backend(self.image_path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return

        self._auto_fit_image()
        self.status.set("Image loaded. Edit image then Set Landmarks.")
        self._update_button_states()

    def _load_image_backend(self, path: Path):
        ext = path.suffix.lower()
        if PIL_OK:
            self.pil_image = Image.open(path).convert("RGBA")
        else:
            if ext not in [".png", ".gif", ".ppm", ".pgm"]:
                raise RuntimeError("This image format needs Pillow. Install: pip install pillow")
            self.pil_image = None
            self.tk_image = tk.PhotoImage(file=str(path))
        self._render_image()

    def _apply_transforms(self, exclude_crop: bool = False):
        if not PIL_OK:
            raise RuntimeError("Transforms require Pillow (pip install pillow).")
        img = self.pil_image
        if self.flip_x:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if abs(self.rotate_deg) > 1e-6:
            img = img.rotate(self.rotate_deg, expand=True, resample=Image.BICUBIC)
        if (not exclude_crop) and self.crop_box is not None:
            img = img.crop(self.crop_box)
        return img

    def on_canvas_resize(self, _evt=None):
        if self.image_path is None:
            return
        self._auto_fit_image()

    def _auto_fit_image(self):
        self.update_idletasks()
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if PIL_OK:
            img = self._apply_transforms(exclude_crop=False)
            w, h = img.size
            self.display_scale = clamp(min(cw / max(w, 1), ch / max(h, 1)), 0.05, 8.0)
        else:
            self.display_scale = 1.0
        self._render_image()

    def _render_image(self):
        self.canvas.delete("all")
        if self.image_path is None:
            return
        if PIL_OK:
            img = self._apply_transforms(exclude_crop=False)
            w, h = img.size
            dw, dh = max(1, int(w * self.display_scale)), max(1, int(h * self.display_scale))
            disp = img.resize((dw, dh), resample=Image.BILINEAR)
            self.tk_image = ImageTk.PhotoImage(disp)
        else:
            self.display_scale = 1.0
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw", tags=("img",))
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._draw_overlays()

    def toggle_flip(self):
        if getattr(self, 'locked_after_save', False):
            return
        if self.image_path is None:
            return
        self.flip_x = not self.flip_x
        self._disable_2d()
        self._auto_fit_image()

    def on_rotate_change(self, _=None):
        if getattr(self, 'locked_after_save', False):
            return
        if self.image_path is None:
            return
        self.rotate_deg = float(self.rot_var.get())
        self._disable_2d()
        self._auto_fit_image()

    def enter_crop(self):
        if getattr(self, 'locked_after_save', False):
            return
        if self.image_path is None:
            return
        self.mode = "crop"
        self.status.set("Crop: drag a rectangle (green). Apply Crop activates after selection.")

    def reset_crop(self):
        if getattr(self, 'locked_after_save', False):
            return
        self.crop_box = None
        self._crop_rect_id = None
        self._drag_start = None
        self._disable_2d()
        if self.btn_apply_crop is not None:
            self.btn_apply_crop.configure(state="disabled")
        self._auto_fit_image()
        self.status.set("Crop reset.")
        self._update_button_states()

    def apply_crop(self):
        if getattr(self, 'locked_after_save', False):
            return
        if self.image_path is None:
            return
        if not PIL_OK:
            messagebox.showwarning("Pillow required", "Cropping requires Pillow. Install: pip install pillow")
            return
        if self.crop_box is None:
            messagebox.showinfo("No crop", "No crop rectangle defined.")
            return
        self.mode = "idle"
        self._disable_2d()
        if self.btn_apply_crop is not None:
            self.btn_apply_crop.configure(state="disabled")
        self._auto_fit_image()
        self.status.set("Crop applied.")
        self._update_button_states()

    def _canvas_to_working(self, x: float, y: float) -> Tuple[float, float]:
        return (x / max(self.display_scale, 1e-9), y / max(self.display_scale, 1e-9))

    def _ray_dirs_from_AB(self):
        if "A" not in self.landmarks or "B" not in self.landmarks:
            return None
        A = self.landmarks["A"]; B = self.landmarks["B"]
        uAB = v_norm(v_sub(A, B))
        def dir_deg(deg):
            return v_norm(rot2(uAB, math.radians(-deg)))
        return {"ray5": dir_deg(5.0), "ray10": dir_deg(10.0), "ray20": dir_deg(20.0), "ray30": dir_deg(30.0)}

    def _ray_dir_D_from_C30(self):
        if "A" not in self.landmarks or "B" not in self.landmarks or "C30" not in self.landmarks:
            return None
        A = self.landmarks["A"]; B = self.landmarks["B"]; C30 = self.landmarks["C30"]
        uAB = v_norm(v_sub(A, B))
        perp = rot2(uAB, math.pi / 2.0)
        if perp[0] < 0:
            perp = v_mul(perp, -1.0)
        return C30, v_norm(perp)

    def _draw_line_on_image(self, img, p0, p1, color, dash=False):
        x0, y0 = p0; x1, y1 = p1
        dx = abs(x1 - x0); dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        step = 0
        while True:
            if not dash or (step // 6) % 2 == 0:
                if 0 <= x0 < img.size[0] and 0 <= y0 < img.size[1]:
                    img.putpixel((x0, y0), color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy; x0 += sx
            if e2 <= dx:
                err += dx; y0 += sy
            step += 1

    def on_mouse_move(self, event):
        if (not PIL_OK) or self.image_path is None:
            return
        img = self._apply_transforms(exclude_crop=False)
        wx, wy = self._canvas_to_working(event.x, event.y)
        x, y = float(wx), float(wy)
        w, h = img.size

        half = self.zoom_half
        x0 = clamp(x - half, 0.0, w - 1.0)
        y0 = clamp(y - half, 0.0, h - 1.0)
        x1 = clamp(x + half, 1.0, float(w))
        y1 = clamp(y + half, 1.0, float(h))

        crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
        zoom = crop.resize((self.zoom_w, self.zoom_h), resample=Image.NEAREST).convert("RGBA")

        def map_pt(px, py):
            zx = (px - x0) / max((x1 - x0), 1e-9) * (self.zoom_w - 1)
            zy = (py - y0) / max((y1 - y0), 1e-9) * (self.zoom_h - 1)
            return int(clamp(zx, 0, self.zoom_w - 1)), int(clamp(zy, 0, self.zoom_h - 1))

        rect = (x0, y0, x1, y1)

        rays = self._ray_dirs_from_AB()
        if rays is not None and "B" in self.landmarks:
            B = self.landmarks["B"]
            for d in rays.values():
                seg = clip_ray_to_rect(B, d, rect)
                if seg:
                    p0, p1 = seg
                    self._draw_line_on_image(zoom, map_pt(*p0), map_pt(*p1), (255,234,0,255), dash=True)

        d_ray = self._ray_dir_D_from_C30()
        if d_ray is not None:
            origin, ddir = d_ray
            seg = clip_ray_to_rect(origin, ddir, rect)
            if seg:
                p0, p1 = seg
                self._draw_line_on_image(zoom, map_pt(*p0), map_pt(*p1), (0,230,118,255), dash=True)

        cx, cy = map_pt(x, y)
        for dx in range(-8, 9):
            zoom.putpixel((int(clamp(cx + dx, 0, self.zoom_w - 1)), cy), (255,64,64,255))
        for dy in range(-8, 9):
            zoom.putpixel((cx, int(clamp(cy + dy, 0, self.zoom_h - 1))), (255,64,64,255))

        self.zoom_imgtk = ImageTk.PhotoImage(zoom)
        self.zoom_canvas.delete("zoomimg")
        self.zoom_canvas.create_image(0, 0, image=self.zoom_imgtk, anchor="nw", tags=("zoomimg",))
        try:
            self.zoom_canvas.tag_raise('zoomimg')
        except Exception:
            pass

    def start_landmarks(self):
        if getattr(self, 'locked_after_save', False):
            return
        if self.image_path is None:
            return
        self.landmarks.clear()
        self.controls = None
        self.mode = "lm_B"
        self._active_handle = None
        self._disable_2d()
        self.status.set("Landmarks: click B.")
        self._render_image()

    def save_measurement(self):
        if self.controls is None:
            messagebox.showwarning("Not ready", "Finish landmarks first (controls appear after D), then adjust Bézier.")
            return
        required = ["B","A","C30","C20","C10","C5","D"]
        if any(k not in self.landmarks for k in required):
            messagebox.showwarning("Missing landmarks", "Some landmarks are missing; restart 'Set Landmarks'.")
            return
        if self._ensure_csv_path() is None:
            return

        lm = Landmarks(
            A=self.landmarks["A"], B=self.landmarks["B"],
            C5=self.landmarks["C5"], C10=self.landmarks["C10"],
            C20=self.landmarks["C20"], C30=self.landmarks["C30"],
            D=self.landmarks["D"]
        )
        ratios = compute_weller_ratios(lm)
        poly = outline_polygon(self.controls, n_per_seg=900)
        area_px2 = shoelace_area(poly)

        Bx, By = lm.B
        def rel(p): return (p[0]-Bx, p[1]-By)

        row: Dict[str, str] = {k: "" for k in self.CSV_COLUMNS}
        # `id` is the image stem (FinShape3D expects this as the primary identifier)
        row["id"] = self.image_path.stem if self.image_path else ""
        row["image_name"] = self.image_path.name if self.image_path else ""
        row["AB_px"] = f"{ratios['AB_px']:.6f}"
        row["area_px2"] = f"{area_px2:.6f}"
        
        # Calculate ratios relative to BC30 for FinShape3D compatibility
        BC30 = dist(lm.C30, lm.B)
        row["r_BA"]   = f"{ratios['AB_px'] / BC30:.10f}"
        row["r_BC5"]  = f"{dist(lm.C5, lm.B) / BC30:.10f}"
        row["r_BC10"] = f"{dist(lm.C10, lm.B) / BC30:.10f}"
        row["r_BC20"] = f"{dist(lm.C20, lm.B) / BC30:.10f}"
        row["r_BC30"] = f"{1.0:.10f}"
        row["DB_over_AB"]   = f"{ratios['DB_over_AB']:.10f}"

        Ax, Ay = rel(lm.A); row["A_dx"], row["A_dy"] = f"{Ax:.6f}", f"{Ay:.6f}"
        x, y = rel(lm.C5);  row["C5_dx"], row["C5_dy"] = f"{x:.6f}", f"{y:.6f}"
        x, y = rel(lm.C10); row["C10_dx"], row["C10_dy"] = f"{x:.6f}", f"{y:.6f}"
        x, y = rel(lm.C20); row["C20_dx"], row["C20_dy"] = f"{x:.6f}", f"{y:.6f}"
        x, y = rel(lm.C30); row["C30_dx"], row["C30_dy"] = f"{x:.6f}", f"{y:.6f}"
        x, y = rel(lm.D);   row["D_dx"], row["D_dy"] = f"{x:.6f}", f"{y:.6f}"

        self.measurements.append(row)
        self._write_csv()
        self.locked_after_save = True
        self._update_button_states()
        self.status.set(f"Saved to CSV ({len(self.measurements)} row(s)). Editing locked for this image.")

    def exit_save(self):
        if self.csv_path is None and self.measurements:
            if self._ensure_csv_path() is None:
                return
        if self.csv_path is not None:
            self._write_csv()
        self.destroy()

    def generate_3d(self):
        if not getattr(self, 'locked_after_save', False):
            messagebox.showinfo('Not ready', 'Save Measurement (CSV) first.')
            return
        if self.controls is None or "B" not in self.landmarks or "A" not in self.landmarks or "C30" not in self.landmarks:
            messagebox.showwarning("Not ready", "Need fitted controls and landmarks B, A, C30.")
            return

        # Prepare data for 3D reconstruction
        lm = Landmarks(
            A=self.landmarks["A"], B=self.landmarks["B"],
            C5=self.landmarks["C5"], C10=self.landmarks["C10"],
            C20=self.landmarks["C20"], C30=self.landmarks["C30"],
            D=self.landmarks["D"]
        )
        ratios = compute_weller_ratios(lm)
        
        # Create a pandas Series with the ratios
        row = pd.Series({
            "C30B_over_AB": ratios["C30B_over_AB"],
            "C5B_over_AB": ratios["C5B_over_AB"],
            "C10B_over_AB": ratios["C10B_over_AB"],
            "C20B_over_AB": ratios["C20B_over_AB"],
            "id": self.image_path.stem if self.image_path else "fin"
        })
        
        # Reconstruct
        from core.reconstruction.wireframe import build_landmarks as build_landmarks_3d
        lm_3d, u_AB = build_landmarks_3d(row, BC30_scale=10.0)
        # Prefer the user-placed AB direction to preserve orientation (flip/rotate)
        # Convert GUI tuple-based vector to numpy for build_controls
        try:
            import numpy as _np
            from core.math.geometry import v_sub, v_norm
            # GUI landmarks are in image pixel coords (y increases downwards).
            # Convert the AB vector to FinShape's y-up convention by flipping y.
            u_ab_gui = v_norm(v_sub(self.landmarks["A"], self.landmarks["B"]))
            u_ab_gui = (float(u_ab_gui[0]), float(-u_ab_gui[1]))
            # Ensure unit-length numpy vector
            arr = _np.array([u_ab_gui[0], u_ab_gui[1]], dtype=float)
            nrm = float(_np.linalg.norm(arr))
            if nrm <= 1e-12:
                raise ValueError("Degenerate AB vector from GUI landmarks")
            u_AB_np = arr / nrm
        except Exception:
            # fallback to the normalized direction returned by build_landmarks
            u_AB_np = u_AB
        try:
            ctrl_3d = build_controls(lm_3d, u_AB_np)
        except Exception as e:
            messagebox.showerror(
                "3D generation failed",
                f"Failed to build 3D controls: {e}\n\n"
                "This may be caused by invalid ratios or a CSV column ordering mismatch.\n"
                f"Row used for reconstruction: {row.to_dict()}"
            )
            return
        curves2d = compute_2d_curves(ctrl_3d, n=900)

        # Plot: save both 2D and 3D outputs to match Eduardo's original pipeline
        outdir = self.image_path.parent / "FinShape3D_outputs"
        outdir.mkdir(parents=True, exist_ok=True)

        # Use the same naming pattern as the standalone FinShape3D script
        try:
            from core.ui import plotting_3d as _plotting
            ver = getattr(_plotting, 'VERSION', '')
        except Exception:
            ver = ''

        fin_id = self.image_path.stem if self.image_path else 'fin'
        p2d = outdir / f"FinShape3D_{ver}_2D_{fin_id}.png" if ver else outdir / f"{fin_id}_2D.png"
        p3d = outdir / f"FinShape3D_{ver}_3D_{fin_id}.png" if ver else outdir / f"{fin_id}_3D.png"

        # 2D contour (annotated) and 3D wireframe
        try:
            # build a landmarks dict expected by plot_2d
            lm_dict = {
                'B': lm_3d['B'], 'A': lm_3d['A'], 'C5': lm_3d['C5'],
                'C10': lm_3d['C10'], 'C20': lm_3d['C20'], 'C30': lm_3d['C30'],
            }
            plot_2d(curves2d, lm_dict, ctrl_3d, fin_id, p2d, legend_anchor_xy=(-4.0, 4.0))
        except Exception as e:
            # Non-fatal: warn but continue to attempt 3D
            messagebox.showwarning("2D plot failed", f"Failed to save 2D contour: {e}")

        try:
            plot_3d_wireframe(curves2d, fin_id, p3d)
        except Exception as e:
            messagebox.showerror("3D generation failed", f"Failed to build 3D wireframe: {e}")
            return

        # Create a small montage of the outputs (2D + 3D) when possible
        try:
            from core.ui.plotting_3d import make_montage
            mont = outdir / f"FinShape3D_{ver}_montage_{fin_id}.png" if ver else outdir / f"{fin_id}_montage.png"
            imgs = [p for p in (p2d, p3d) if p.exists()]
            if imgs:
                make_montage(imgs, mont, title=f"FinShape3D {ver} — {fin_id}" if ver else None)
        except Exception:
            mont = None

        # Open the 3D image (previous behaviour) or montage if available
        to_open = mont if (mont and mont.exists()) else p3d
        try:
            if platform.system() == "Darwin":
                subprocess.call(["open", str(to_open)])
            elif platform.system() == "Windows":
                os.startfile(str(to_open))
            else:
                opener = shutil.which("xdg-open")
                if opener:
                    subprocess.call([opener, str(to_open)])
                else:
                    try:
                        webbrowser.open(to_open.as_uri())
                    except Exception:
                        messagebox.showinfo("3D generated", f"Saved: {to_open}\nOpen it manually.")
        except FileNotFoundError:
            messagebox.showinfo("Open failed", f"Saved: {to_open}\nCould not open automatically (xdg-open not found).")

        self.status.set(f"3D generated: {p3d.name}")

    def show_2d_window(self):
        if not getattr(self, 'locked_after_save', False):
            messagebox.showinfo('Not ready', 'Save Measurement (CSV) first.')
            return
        if self.controls is None or "B" not in self.landmarks or "A" not in self.landmarks or "C30" not in self.landmarks:
            messagebox.showwarning("Not ready", "Need fitted controls and landmarks B, A, C30.")
            return

        poly = outline_polygon(self.controls, n_per_seg=1100)
        B = self.landmarks["B"]
        A = self.landmarks["A"]
        C30 = self.landmarks["C30"]

        def rel_px(p):
            # relative to B in image pixel coords (x right, y down)
            return (p[0] - B[0], p[1] - B[1])

        def to_yup(p):
            # convert to y-up coords
            return (p[0], -p[1])

        poly_rel_yup = [to_yup(rel_px(p)) for p in poly]
        A_rel_yup = to_yup(rel_px(A))
        C30_rel_yup = to_yup(rel_px(C30))

        # rotate in y-up so B->C30 becomes +x
        ang = -math.atan2(C30_rel_yup[1], C30_rel_yup[0])
        poly_rot = [rotate_about_origin(p, ang) for p in poly_rel_yup]
        A_rot = rotate_about_origin(A_rel_yup, ang)

        # ensure fin swims right
        if A_rot[0] < 0:
            poly_rot = [(-p[0], p[1]) for p in poly_rot]
            A_rot = (-A_rot[0], A_rot[1])

        AB_px = v_len(A_rot)
        if AB_px < 1e-9:
            messagebox.showwarning("Invalid", "AB length is zero.")
            return
        scale = 10.0 / AB_px

        # final units: x right, y up
        poly_units = [(p[0] * scale, p[1] * scale) for p in poly_rot]        # User-required display transform:
        # Flip the contour over the OTHER axis for display (mirror vertically):
        # y -> -y (so y increases upward and the fin is not upside down)
        poly_units = [(x, -y) for (x, y) in poly_units]
        win = tk.Toplevel(self)
        win.title("2D Contour (FinShape2D R-GUI 2.0.13)")
        win.geometry("980x880")

        top = tk.Frame(win, bd=1, relief="raised"); top.pack(side="top", fill="x")
        tk.Label(top,
                 text=f"{self.image_path.name if self.image_path else ''} | Rotated: B→C30 horizontal | Scaled: |AB|=10",
                 anchor="w").pack(side="left", padx=8, pady=4)

        tk.Button(top, text="Save contour…", command=lambda: self.save_contour_dialog(poly_units)).pack(side="right", padx=8, pady=4)

        cv = tk.Canvas(win, bg="white"); cv.pack(side="top", fill="both", expand=True)

        def fit_points_yup(pts, W, H, pad=90):
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
            sx = (W - 2*pad) / max(maxx - minx, 1e-9)
            sy = (H - 2*pad) / max(maxy - miny, 1e-9)
            sc = min(sx, sy)

            def tx(p):
                x, y = p
                cx = pad + (x - minx)*sc
                cy = pad + (maxy - y)*sc
                return (cx, cy)

            return [tx(p) for p in pts], (minx, miny, maxx, maxy, sc, pad)

        def draw_axes(W, H, bbox):
            minx, miny, maxx, maxy, sc, pad = bbox
            ox = pad + (0.0 - minx) * sc
            oy = pad + (maxy - 0.0) * sc
            cv.create_line(0, oy, W, oy, fill="#666", width=1)
            cv.create_line(ox, 0, ox, H, fill="#666", width=1)

            tick = 1.0
            k0 = int(math.floor(minx / tick)) - 1
            k1 = int(math.ceil(maxx / tick)) + 1
            for k in range(k0, k1 + 1):
                x = pad + (k*tick - minx) * sc
                cv.create_line(x, oy-4, x, oy+4, fill="#666", width=1)
                if k % 2 == 0:
                    cv.create_text(x, oy+14, text=str(k), fill="#444", font=("Helvetica", 9))

            k0 = int(math.floor(miny / tick)) - 1
            k1 = int(math.ceil(maxy / tick)) + 1
            for k in range(k0, k1 + 1):
                y = pad + (maxy - k*tick) * sc
                cv.create_line(ox-4, y, ox+4, y, fill="#666", width=1)
                if k % 2 == 0:
                    cv.create_text(ox-14, y, text=str(k), fill="#444", font=("Helvetica", 9))

            cv.create_text(W-10, oy-10, anchor="ne", text="x (AB=10 units)", fill="#444", font=("Helvetica", 10, "bold"))
            cv.create_text(ox+10, 10, anchor="nw", text="y (AB=10 units)", fill="#444", font=("Helvetica", 10, "bold"))

        def redraw(_=None):
            cv.delete("all")
            W = max(1, cv.winfo_width()); H = max(1, cv.winfo_height())
            pts, bbox = fit_points_yup(poly_units, W, H, pad=90)
            draw_axes(W, H, bbox)
            for i in range(len(pts)-1):
                cv.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill="black", width=2)
            cv.create_line(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], fill="black", width=2)

        cv.bind("<Configure>", redraw)
        redraw()

    def save_contour_dialog(self, poly_units: List[Tuple[float, float]]):
        """Save the 2D contour. SVG is always supported; PNG/JPG/BMP require Pillow but do NOT require Ghostscript."""
        stem = self.image_path.stem if self.image_path else "contour"
        filetypes = [("SVG", "*.svg")]
        if PIL_OK:
            filetypes += [("PNG", "*.png"), ("JPG", "*.jpg"), ("BMP", "*.bmp")]
        filetypes += [("PostScript", "*.ps")]

        out = filedialog.asksaveasfilename(
            title="Save contour image",
            initialfile=f"{stem}_contour.svg",
            defaultextension=".svg",
            filetypes=filetypes,
        )
        if not out:
            return
        out_path = Path(out)
        ext = out_path.suffix.lower()

        if ext == ".svg":
            self._save_svg(poly_units, out_path)
            messagebox.showinfo("Saved", f"Saved: {out_path.name}")
            return

        if ext in [".png", ".jpg", ".bmp"]:
            if not PIL_OK:
                messagebox.showwarning("Pillow missing", "Raster export requires Pillow (pip install pillow).")
                return
            self._save_raster_contour(poly_units, out_path, ext)
            messagebox.showinfo("Saved", f"Saved: {out_path.name}")
            return

        # Vector PostScript
        self._save_ps_contour(poly_units, out_path)
        messagebox.showinfo("Saved", f"Saved: {out_path.name}")

    def _save_raster_contour(self, poly_units: List[Tuple[float, float]], out_path: Path, ext: str):
        """Direct raster rendering using Pillow (no PostScript/Ghostscript)."""
        W, H = 1400, 1400
        pad = 140
        img = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(img)

        xs = [p[0] for p in poly_units]; ys = [p[1] for p in poly_units]
        minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
        sx = (W - 2*pad) / max(maxx - minx, 1e-9)
        sy = (H - 2*pad) / max(maxy - miny, 1e-9)
        sc = min(sx, sy)

        def tx(p):
            x, y = p
            cx = pad + (x - minx) * sc
            cy = pad + (maxy - y) * sc  # y-up to canvas
            return (cx, cy)

        pts = [tx(p) for p in poly_units]

        ox = pad + (0.0 - minx) * sc
        oy = pad + (maxy - 0.0) * sc
        dr.line([(0, oy), (W, oy)], fill=(90, 90, 90), width=3)
        dr.line([(ox, 0), (ox, H)], fill=(90, 90, 90), width=3)

        for i in range(len(pts)-1):
            dr.line([pts[i], pts[i+1]], fill=(0, 0, 0), width=4)
        dr.line([pts[-1], pts[0]], fill=(0, 0, 0), width=4)

        fmt = "PNG" if ext == ".png" else ("JPEG" if ext == ".jpg" else "BMP")
        img.save(str(out_path), fmt)

    def _save_ps_contour(self, poly_units: List[Tuple[float, float]], out_path: Path):
        """Vector PostScript export using Tk canvas."""
        W, H = 1200, 1200
        pad = 120
        tmp = tk.Toplevel(self)
        tmp.withdraw()
        cv = tk.Canvas(tmp, width=W, height=H, bg="white")
        cv.pack()

        xs = [p[0] for p in poly_units]; ys = [p[1] for p in poly_units]
        minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
        sx = (W - 2*pad) / max(maxx - minx, 1e-9)
        sy = (H - 2*pad) / max(maxy - miny, 1e-9)
        sc = min(sx, sy)

        def tx(p):
            x, y = p
            cx = pad + (x - minx) * sc
            cy = pad + (maxy - y) * sc
            return (cx, cy)

        pts = [tx(p) for p in poly_units]

        ox = pad + (0.0 - minx) * sc
        oy = pad + (maxy - 0.0) * sc
        cv.create_line(0, oy, W, oy, fill="#666", width=3)
        cv.create_line(ox, 0, ox, H, fill="#666", width=3)
        for i in range(len(pts)-1):
            cv.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill="black", width=4)
        cv.create_line(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], fill="black", width=4)

        cv.postscript(file=str(out_path), colormode="color")
        tmp.destroy()

    def _save_svg(self, poly_units: List[Tuple[float, float]], out_path: Path):
        W, H = 1000, 1000
        pad = 100
        xs = [p[0] for p in poly_units]; ys = [p[1] for p in poly_units]
        minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
        sx = (W - 2*pad) / max(maxx - minx, 1e-9)
        sy = (H - 2*pad) / max(maxy - miny, 1e-9)
        sc = min(sx, sy)

        def tx(p):
            x, y = p
            cx = pad + (x - minx)*sc
            cy = pad + (maxy - y)*sc
            return (cx, cy)

        pts = [tx(p) for p in poly_units]
        pts_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in pts])

        ox = pad + (0.0 - minx) * sc
        oy = pad + (maxy - 0.0) * sc

        svg = []
        svg.append('<?xml version="1.0" encoding="UTF-8"?>')
        svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
        svg.append('<rect width="100%" height="100%" fill="white"/>')
        svg.append(f'<line x1="0" y1="{oy:.2f}" x2="{W}" y2="{oy:.2f}" stroke="#666" stroke-width="2"/>')
        svg.append(f'<line x1="{ox:.2f}" y1="0" x2="{ox:.2f}" y2="{H}" stroke="#666" stroke-width="2"/>')
        svg.append(f'<polyline points="{pts_str}" fill="none" stroke="black" stroke-width="3"/>')
        svg.append(f'<line x1="{pts[-1][0]:.2f}" y1="{pts[-1][1]:.2f}" x2="{pts[0][0]:.2f}" y2="{pts[0][1]:.2f}" stroke="black" stroke-width="3"/>')
        svg.append('</svg>')
        out_path.write_text("\n".join(svg), encoding="utf-8")

    def on_mouse_down(self, event):
        if self.image_path is None:
            return
        x, y = event.x, event.y

        if self.mode == "crop":
            self._drag_start = (x, y)
            if self._crop_rect_id is not None:
                self.canvas.delete(self._crop_rect_id)
            self._crop_rect_id = self.canvas.create_rectangle(x, y, x, y, outline="#00e676", width=2)
            return

        if self.controls is not None and self.mode == "bezier":
            item = self.canvas.find_withtag("current")
            if item:
                tags = self.canvas.gettags(item)
                for t in tags:
                    if t in self.controls.__dataclass_fields__:
                        self._active_handle = t
                        self._drag_start = (x, y)
                        return

        wx, wy = self._canvas_to_working(x, y)

        if self.mode == "lm_B":
            self.landmarks["B"] = (wx, wy); self.mode = "lm_A"; self.status.set("Landmarks: click A."); self._render_image()
        elif self.mode == "lm_A":
            self.landmarks["A"] = (wx, wy); self.mode = "lm_C30"; self.status.set("Landmarks: click C30."); self._render_image()
        elif self.mode == "lm_C30":
            self.landmarks["C30"] = (wx, wy); self.mode = "lm_C20"; self.status.set("Landmarks: click C20."); self._render_image()
        elif self.mode == "lm_C20":
            self.landmarks["C20"] = (wx, wy); self.mode = "lm_C10"; self.status.set("Landmarks: click C10."); self._render_image()
        elif self.mode == "lm_C10":
            self.landmarks["C10"] = (wx, wy); self.mode = "lm_C5"; self.status.set("Landmarks: click C5."); self._render_image()
        elif self.mode == "lm_C5":
            self.landmarks["C5"] = (wx, wy); self.mode = "lm_D"; self.status.set("Landmarks: click D."); self._render_image()
        elif self.mode == "lm_D":
            self.landmarks["D"] = (wx, wy)
            lm = Landmarks(
                A=self.landmarks["A"], B=self.landmarks["B"],
                C5=self.landmarks["C5"], C10=self.landmarks["C10"],
                C20=self.landmarks["C20"], C30=self.landmarks["C30"],
                D=self.landmarks["D"]
            )
            self.controls = build_initial_controls(lm)
            self.mode = "bezier"
            self.status.set("Landmarks complete. Drag orange control points, then Save Measurement (CSV).")
            self._render_image()

    def on_mouse_drag(self, event):
        if self.image_path is None:
            return
        x, y = event.x, event.y

        if self.mode == "crop" and self._drag_start is not None and self._crop_rect_id is not None:
            x0, y0 = self._drag_start
            self.canvas.coords(self._crop_rect_id, x0, y0, x, y)
            return

        if self.controls is not None and self._active_handle is not None and self._drag_start is not None:
            dx = (x - self._drag_start[0]) / max(self.display_scale, 1e-9)
            dy = (y - self._drag_start[1]) / max(self.display_scale, 1e-9)
            self._drag_start = (x, y)
            p = getattr(self.controls, self._active_handle)
            setattr(self.controls, self._active_handle, (p[0] + dx, p[1] + dy))
            self._disable_2d()
            self._render_image()

    def on_mouse_up(self, event):
        if self.image_path is None:
            return

        if self.mode == "crop" and self._crop_rect_id is not None:
            if not PIL_OK:
                return
            img_t = self._apply_transforms(exclude_crop=True)
            tw, th = img_t.size
            x0, y0, x1, y1 = self.canvas.coords(self._crop_rect_id)
            sx = 1.0 / max(self.display_scale, 1e-9)
            x0i, y0i = int(min(x0, x1) * sx), int(min(y0, y1) * sx)
            x1i, y1i = int(max(x0, x1) * sx), int(max(y0, y1) * sx)

            x0i = int(clamp(x0i, 0, tw - 1)); y0i = int(clamp(y0i, 0, th - 1))
            x1i = int(clamp(x1i, 1, tw));     y1i = int(clamp(y1i, 1, th))

            if abs(x1i - x0i) > 5 and abs(y1i - y0i) > 5:
                self.crop_box = (x0i, y0i, x1i, y1i)
                if self.btn_apply_crop is not None:
                    self.btn_apply_crop.configure(state="normal")
                self._update_button_states()
                self.status.set("Crop box set. Click 'Apply Crop' to finalize.")
            self._drag_start = None
            return

        self._active_handle = None
        self._drag_start = None

    def _draw_overlays(self):
        for k, p in self.landmarks.items():
            cx, cy = p[0] * self.display_scale, p[1] * self.display_scale
            r = 5
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#00e5ff", outline="")
            self.canvas.create_text(cx + 10, cy - 10, text=k, fill="#00e5ff")

        rays = self._ray_dirs_from_AB()
        if rays is not None and "B" in self.landmarks:
            B = self.landmarks["B"]; length = 2400
            for d in rays.values():
                x0, y0 = B; x1, y1 = x0 + d[0] * length, y0 + d[1] * length
                self.canvas.create_line(x0 * self.display_scale, y0 * self.display_scale,
                                        x1 * self.display_scale, y1 * self.display_scale,
                                        fill="#ffea00", width=1, dash=(4, 4))

        d_ray = self._ray_dir_D_from_C30()
        if d_ray is not None:
            origin, ddir = d_ray; length = 2400
            x0, y0 = origin; x1, y1 = x0 + ddir[0] * length, y0 + ddir[1] * length
            col = "#00e676" if self.mode == "lm_D" else "#00c853"
            self.canvas.create_line(x0 * self.display_scale, y0 * self.display_scale,
                                    x1 * self.display_scale, y1 * self.display_scale,
                                    fill=col, width=2 if self.mode == "lm_D" else 1, dash=(6, 4))

        if self.controls is not None:
            for field in self.controls.__dataclass_fields__.keys():
                p = getattr(self.controls, field)
                cx, cy = p[0] * self.display_scale, p[1] * self.display_scale
                r = 4
                self.canvas.create_rectangle(cx - r, cy - r, cx + r, cy + r, fill="#ff9800", outline="", tags=("handle", field))
            # Draw a single continuous outline to ensure segment endpoints match exactly.
            poly = outline_polygon(self.controls, n_per_seg=600)
            for i in range(len(poly) - 1):
                x0, y0 = poly[i]; x1, y1 = poly[i + 1]
                self.canvas.create_line(x0 * self.display_scale, y0 * self.display_scale,
                                        x1 * self.display_scale, y1 * self.display_scale,
                                        fill="black", width=2)


def main():
    app = FinShape2DApp()
    if not PIL_OK:
        messagebox.showwarning(
            "Optional dependency missing",
            "Pillow is not installed.\n\n"
            "You can still load PNG/GIF/PPM/PGM.\n"
            "Flip/Rotate/Crop accuracy + Zoom + JPEG require Pillow.\n\n"
            "Install Pillow:\n    pip install pillow"
        )
    app.mainloop()


if __name__ == "__main__":
    main()
