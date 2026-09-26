#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinShape Surfer GUI 1.1.3

Purpose
-------
Integrates the latest versioned FinShape2D and FinShape3D prototypes without
rewriting their scientific formulas. The GUI keeps the FinShape2D image
measurement workflow and, after a successful measurement save, automatically
creates a versioned output folder containing:

- measured 2D contour SVG from the fitted image contour;
- fitted measured 2D contour PNG using the same standardized outline sent to 3D;
- 3D wireframe PNG using the fitted measured 2D contour plus the FinShape3D section/thickness pipeline;
- rotatable embedded 3D view with pitch, yaw, and roll controls after output generation;
- text manifest with source versions, input image, measurement CSV, warnings,
  and exact output paths.

Surface preparation in 1.1.3
--------------------------
An explicit action in the 3D window exports an experimental closed STL/OBJ
surface with physical scaling, validation and incomplete simulation templates.
It requires acceptance of modeled thickness. It does not execute CFD or bioheat.
"""
from __future__ import annotations

import importlib.util
import base64
import io
import copy
from dataclasses import asdict
import hashlib
import os
import re
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

INTEGRATED_VERSION = "1.1.3"
CODE_DIR = Path(__file__).resolve().parent
FINSHAPE2D_PATH = CODE_DIR / "FinShape2D_R-GUI_2.0.20.py"
FINSHAPE3D_PATH = CODE_DIR / "FinShape3D_1.1.20.py"
MEASUREMENT_EXAMPLE_PATH = CODE_DIR / "assets" / "dorsal_fin_tricolor_otsu.png"
SURFACE_EXPORT_PATH = CODE_DIR / "FinShapeCFD_0.1.0.py"

# Exact user-supplied PNG, retained for installations missing the assets folder.
MEASUREMENT_EXAMPLE_PNG_BASE64 = (
    "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"
    "OPkPcJn5OzhuiZsAAAAASUVORK5CYII="
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


fin2d = _load_module("finshape2d_r_gui_2_0_20", FINSHAPE2D_PATH)


def sanitize_output_id(raw: str, fallback: str = "row0") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def output_dir_for_measurement(csv_path: Optional[Path], image_path: Optional[Path]) -> Path:
    if image_path is not None:
        base_dir = image_path.expanduser().resolve().parent
        stem = image_path.stem
    elif csv_path is not None:
        base_dir = csv_path.expanduser().resolve().parent
        stem = csv_path.stem
    else:
        base_dir = Path.cwd().resolve()
        stem = "FinShape"

    return base_dir / f"{sanitize_output_id(stem)}_FinShape2D3D_outputs_v{INTEGRATED_VERSION}"


def measurement_row_to_3d_series(row: Dict[str, str], fin3d_module, fallback_id: str):
    image_name = row.get("image_name", "")
    source_id = Path(image_name).stem if image_name else fallback_id
    data = {
        "id": sanitize_output_id(source_id, fallback=fallback_id),
        "C5B_over_AB": float(row["C5B_over_AB"]),
        "C10B_over_AB": float(row["C10B_over_AB"]),
        "C20B_over_AB": float(row["C20B_over_AB"]),
        "C30B_over_AB": float(row["C30B_over_AB"]),
    }
    return fin3d_module.pd.Series(data)


def surface_setup_message(error):
    """Report the interpreter actually running the GUI; never install on import."""
    command_args = [
        sys.executable, "-m", "pip", "install", "--only-binary=:all:",
        "shapely==2.1.2", "trimesh==4.11.2",
    ]
    command = subprocess.list2cmdline(command_args) if os.name == "nt" else shlex.join(command_args)
    return (
        "The surface-export requirements are missing or incompatible in this Python environment.\n\n"
        f"Running interpreter:\n{sys.executable}\n\n"
        f"Details: {type(error).__name__}: {error}\n\n"
        "Install the tested surface packages with this exact command:\n"
        f"{command}\n\n"
        "Then restart Surfer with the same interpreter. No packages were installed automatically. "
        "Your saved measurement and 2D/3D outputs are unchanged."
    )


class IntegratedFinShapeApp(fin2d.FinShape2DApp):
    def __init__(self):
        print(f"[STARTUP] FinShape Surfer GUI {INTEGRATED_VERSION}")
        print(f"[STARTUP] Python interpreter: {sys.executable}")
        print(f"[STARTUP] 2D source: {FINSHAPE2D_PATH}")
        print(f"[STARTUP] 3D source: {FINSHAPE3D_PATH}")
        print("[STARTUP] Measurement CSV defaults to the loaded image folder.")
        print("[STARTUP] Exact integrated output directory will be printed after Save Measurement.")
        self.source_image_path: Optional[Path] = None
        self.source_image_dir: Optional[Path] = None
        self._button_normal_bg = None
        self._active_button_key: Optional[str] = None
        self._button_registry = {}
        self.measurement_example_imgtk = None
        super().__init__()
        self.title(f"FinShape Surfer GUI {INTEGRATED_VERSION}")
        self._fin3d_module = None
        self.last_integrated_3d_path: Optional[Path] = None
        self.last_integrated_curves2d = None
        self.last_integrated_fin_id: Optional[str] = None
        self.last_3d_view_figure = None
        self.last_integrated_surface_snapshot = None
        self._add_toolbar_divider()
        self._configure_compact_workflow_buttons()
        self.btn_show3d = self._add_show3d_button()
        self._init_active_button_feedback()
        self._replace_instruction_panel()

    def _add_show3d_button(self):
        parent = getattr(getattr(self, "btn_show2d", None), "master", None)
        if parent is None:
            return None
        btn = fin2d.tk.Button(parent, text="9. Show 3D", command=self.show_3d_window, state="disabled")
        try:
            btn.pack(side="left", padx=4, pady=3, before=self.btn_new_image)
        except Exception:
            btn.pack(side="left", padx=4, pady=3)
        return btn

    def _configure_compact_workflow_buttons(self) -> None:
        if getattr(self, "btn_reset_crop", None) is not None:
            try:
                self.btn_reset_crop.pack_forget()
                self.btn_reset_crop.destroy()
            except Exception:
                pass
            self.btn_reset_crop = None
        if getattr(self, "btn_crop", None) is not None:
            self.btn_crop.configure(text="4. Crop", command=self.enter_crop)
        if getattr(self, "btn_apply_crop", None) is not None:
            self.btn_apply_crop.configure(text="5. Apply Crop")
        if getattr(self, "btn_set_landmarks", None) is not None:
            self.btn_set_landmarks.configure(text="6. Set Landmarks", command=self.start_landmarks)
        if getattr(self, "btn_save_meas", None) is not None:
            self.btn_save_meas.configure(text="7. Save Measurement (CSV)")
        if getattr(self, "btn_show2d", None) is not None:
            self.btn_show2d.configure(text="8. Show 2D")
        if getattr(self, "btn_new_image", None) is not None:
            self.btn_new_image.configure(text="10. Measure new image")

    def _add_toolbar_divider(self) -> None:
        row2 = getattr(getattr(self, "btn_show2d", None), "master", None)
        bar = getattr(row2, "master", None)
        if row2 is None or bar is None:
            return
        row2.pack_forget()
        sep = fin2d.tk.Frame(bar, height=1, bg="#8a8a8a")
        sep.pack(side="top", fill="x", padx=6, pady=(1, 1))
        row2.pack(side="top", fill="x")
        self.toolbar_divider = sep

    def _replace_instruction_panel(self) -> None:
        if getattr(self, "info", None) is None:
            return
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert(
            "end",
            "Instructions\n\n"
            "• For consistent measurements, orient every fin so the animal is swimming to the right before landmarking.\n"
            "• Zoom (upper-right): move mouse over image to see 5× zoom with stable rays.\n"
            "• IMPORTANT: click using the TIP of the mouse cursor.\n\n"
            "Edit image: Flip / Rotate / Crop.\n"
            "  Apply Crop activates only after you draw a crop box.\n\n"
            "Morphometry:\n"
            "  Landmarks: B → A → C30 → C20 → C10 → C5 → D\n"
            "  After D, Bézier control points appear automatically.\n"
            "  Drag green control points to fit contour.\n"
            "  Save Measurement updates the session CSV.\n"
            "  Show 2D displays the fitted contour with axes (AB=10).\n"
            "  Show 3D opens the rotatable wireframe and surface export.\n\n"
            "Measurement example:\n"
        )
        self._insert_measurement_example_image()
        self.info.insert(
            "end",
            "\nSource: Morteo et al., 2017. Reference: Morteo E., Rocha-Olivares A., "
            "Morteo R., Weller D.W. 2017. Phenotypic variation in dorsal fin "
            "morphology of coastal bottlenose dolphins (Tursiops truncatus) off Mexico. "
            "PeerJ 5:e3415. ISSN 2167-8359, doi.org/10.7717/peerj.3415.\n\n"
            "Integrated output:\n"
            "  After Save Measurement, this version writes measured 2D SVG, measured 2D PNG, "
            "3D wireframe PNG, and a provenance manifest.\n"
        )
        self.info.configure(state="disabled")

    def _insert_measurement_example_image(self) -> None:
        if not fin2d.PIL_OK:
            original = fin2d.tk.PhotoImage(data=MEASUREMENT_EXAMPLE_PNG_BASE64)
            factor = max(1, (original.width() + 329) // 330)
            self.measurement_example_imgtk = original.subsample(factor, factor)
        else:
            try:
                with fin2d.Image.open(MEASUREMENT_EXAMPLE_PATH) as source:
                    img = source.convert("RGB")
            except (OSError, ValueError):
                with fin2d.Image.open(io.BytesIO(base64.b64decode(MEASUREMENT_EXAMPLE_PNG_BASE64))) as source:
                    img = source.convert("RGB")
            scale = min(330 / max(img.width, 1), 1.0)
            size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(size, resample=fin2d.Image.BILINEAR)
            self.measurement_example_imgtk = fin2d.ImageTk.PhotoImage(img)
        self.info.image_create("end", image=self.measurement_example_imgtk)
        self.info.insert("end", "\n")

    def _init_active_button_feedback(self) -> None:
        self._button_registry = {
            "load": self._find_button_by_prefix("1."),
            "flip": getattr(self, "btn_flip", None),
            "crop": getattr(self, "btn_crop", None),
            "apply_crop": getattr(self, "btn_apply_crop", None),
            "landmarks": getattr(self, "btn_set_landmarks", None),
            "save": getattr(self, "btn_save_meas", None),
            "show2d": getattr(self, "btn_show2d", None),
            "show3d": getattr(self, "btn_show3d", None),
            "new": getattr(self, "btn_new_image", None),
        }
        sample = next((b for b in self._button_registry.values() if b is not None), None)
        if sample is not None:
            self._button_normal_bg = sample.cget("background")
        self._apply_active_button_style()

    def _find_button_by_prefix(self, prefix: str):
        stack = [self]
        while stack:
            widget = stack.pop()
            try:
                children = widget.winfo_children()
            except Exception:
                continue
            stack.extend(children)
            for child in children:
                try:
                    if isinstance(child, fin2d.tk.Button) and str(child.cget("text")).startswith(prefix):
                        return child
                except Exception:
                    pass
        return None

    def _set_active_button(self, key: Optional[str]) -> None:
        self._active_button_key = key
        self._apply_active_button_style()

    def _apply_active_button_style(self) -> None:
        normal_bg = self._button_normal_bg or "SystemButtonFace"
        for key, button in getattr(self, "_button_registry", {}).items():
            if button is None:
                continue
            color = "#b8b8b8" if key == self._active_button_key else normal_bg
            try:
                button.configure(background=color, activebackground=color)
            except Exception:
                pass

    def _sync_active_button_to_mode(self) -> None:
        mode = getattr(self, "mode", "idle")
        if mode == "crop":
            self._set_active_button("crop")
        elif str(mode).startswith("lm_") or mode == "bezier":
            self._set_active_button("landmarks")
        elif getattr(self, "locked_after_save", False):
            self._set_active_button(None)
        else:
            self._set_active_button(None)

    def load_image(self):
        self._set_active_button("load")
        super().load_image()
        if getattr(self, "image_path", None) is not None:
            self._remember_source_image_directory()
            print(f"[INFO] Source image directory remembered: {self.source_image_dir}")
        self._sync_active_button_to_mode()

    def _load_image_backend(self, path):
        super()._load_image_backend(path)
        self._clear_integrated_output_state()
        self.landmark_placement_notes = []
        self.btn_set_landmarks.configure(text="6. Set Landmarks")
        self.btn_crop.configure(text="4. Crop", command=self.enter_crop)

    def toggle_flip(self):
        self._set_active_button("flip")
        super().toggle_flip()
        self.after(350, self._sync_active_button_to_mode)

    def on_rotate_change(self, event=None):
        self._set_active_button(None)
        super().on_rotate_change(event)

    def enter_crop(self):
        if getattr(self, "crop_box", None) is not None and getattr(self, "mode", "idle") != "crop":
            self.reset_crop()
            return
        super().enter_crop()
        if getattr(self, "btn_crop", None) is not None:
            self.btn_crop.configure(text="4. Crop")
        self._sync_active_button_to_mode()

    def apply_crop(self):
        self._set_active_button("apply_crop")
        super().apply_crop()
        if getattr(self, "crop_box", None) is not None and getattr(self, "btn_crop", None) is not None:
            self.btn_crop.configure(text="4. Reset Crop", command=self.reset_crop)
        self.after(350, self._sync_active_button_to_mode)

    def reset_crop(self):
        self._set_active_button("crop")
        super().reset_crop()
        if getattr(self, "btn_crop", None) is not None:
            self.btn_crop.configure(text="4. Crop", command=self.enter_crop)
        self.after(350, self._sync_active_button_to_mode)

    def start_landmarks(self):
        if self.image_path is None or self.locked_after_save:
            return
        if self._landmark_workflow_active():
            self.reset_landmarks()
            return
        super().start_landmarks()
        if getattr(self, "btn_set_landmarks", None) is not None:
            self.btn_set_landmarks.configure(text="6. Reset Landmarks")
        self._update_button_states()
        self._sync_active_button_to_mode()

    def _landmark_workflow_active(self) -> bool:
        mode = getattr(self, "mode", "idle")
        return (str(mode).startswith("lm_") or mode == "bezier" or bool(getattr(self, "landmarks", {})) or getattr(self, "controls", None) is not None) and not getattr(self, "locked_after_save", False)

    def _clear_integrated_output_state(self):
        self.last_integrated_3d_path = None
        self.last_integrated_curves2d = None
        self.last_integrated_fin_id = None
        self.last_integrated_surface_snapshot = None
        self._set_show3d_enabled(False)

    def reset_landmarks(self) -> None:
        if self.locked_after_save or self.image_path is None:
            return
        self._clear_integrated_output_state()
        super().start_landmarks()
        self.btn_set_landmarks.configure(text="6. Reset Landmarks")
        self._update_button_states()
        self._sync_active_button_to_mode()
        self.status.set("Landmarks reset. Click B to begin again.")

    def _update_button_states(self):
        super()._update_button_states()
        loaded = self.image_path is not None
        editing_landmarks = bool(self.landmarks) or self.controls is not None
        can_edit_image = loaded and not self.locked_after_save and not editing_landmarks
        for name in ("btn_flip", "btn_crop", "scale_rotate"):
            button = self.__dict__.get(name)
            if button is not None:
                button.configure(state="normal" if can_edit_image else "disabled")
        self.btn_apply_crop.configure(
            state="normal" if can_edit_image and self.mode == "crop" and self.crop_box else "disabled",
        )
        self.btn_set_landmarks.configure(state="normal" if loaded and not self.locked_after_save else "disabled")
        self.btn_save_meas.configure(state="normal" if self.controls is not None and not self.locked_after_save else "disabled")

    def on_mouse_down(self, event):
        super().on_mouse_down(event)
        self._update_button_states()

    def measure_new_image(self):
        self._set_active_button("new")
        self._clear_integrated_output_state()
        if getattr(self, "btn_crop", None) is not None:
            self.btn_crop.configure(text="4. Crop", command=self.enter_crop)
        if getattr(self, "btn_set_landmarks", None) is not None:
            self.btn_set_landmarks.configure(text="6. Set Landmarks")
        super().measure_new_image()
        self._sync_active_button_to_mode()

    def _set_show3d_enabled(self, enabled: bool) -> None:
        if getattr(self, "btn_show3d", None) is not None:
            self.btn_show3d.configure(state=("normal" if enabled else "disabled"))
            self._apply_active_button_style()

    def _remember_source_image_directory(self) -> Optional[Path]:
        image_path = self.__dict__.get("image_path")
        if image_path is not None:
            candidate = Path(image_path).expanduser()
            if candidate.exists():
                self.source_image_path = candidate.resolve()
                self.source_image_dir = self.source_image_path.parent
                return self.source_image_dir
        remembered = self.__dict__.get("source_image_dir")
        if remembered is not None:
            remembered_path = Path(remembered).expanduser()
            if remembered_path.exists():
                self.source_image_dir = remembered_path.resolve()
                return self.source_image_dir
        return None

    def _source_save_directory(self) -> Path:
        initial_dir = self._remember_source_image_directory() or Path.cwd().resolve()
        initial_dir = Path(initial_dir).expanduser()
        if initial_dir.exists():
            return initial_dir.resolve()
        return Path.cwd().resolve()

    def _ask_save_file_in_source_dir(self, *, title: str, default_name: str, defaultextension: str, filetypes, source_directory=None) -> Optional[Path]:
        initial_dir = Path(source_directory).expanduser().resolve() if source_directory is not None else self._source_save_directory()
        default_filename = Path(default_name).name
        print(f"[INFO] Save dialog initial directory: {initial_dir}")
        previous_cwd = Path.cwd()
        previous_tk_pwd = None
        try:
            tk_obj = self.__dict__.get("tk")
            try:
                if tk_obj is not None:
                    previous_tk_pwd = tk_obj.call("pwd")
                    tk_obj.call("cd", str(initial_dir))
            except Exception:
                previous_tk_pwd = None
            os.chdir(initial_dir)
            selected = fin2d.filedialog.asksaveasfilename(
                parent=self,
                title=title,
                initialdir=str(initial_dir),
                initialfile=default_filename,
                defaultextension=defaultextension,
                filetypes=filetypes,
                confirmoverwrite=True,
            )
        finally:
            if previous_tk_pwd is not None:
                try:
                    tk_obj = self.__dict__.get("tk")
                    if tk_obj is not None:
                        tk_obj.call("cd", previous_tk_pwd)
                except Exception:
                    pass
            try:
                os.chdir(previous_cwd)
            except Exception:
                pass
        if not selected:
            return None
        out_path = Path(selected).expanduser()
        if not out_path.is_absolute():
            out_path = initial_dir / out_path
        return out_path

    def _ensure_csv_path(self) -> Optional[Path]:
        if self.csv_path is not None:
            return self.csv_path
        out_path = self._ask_save_file_in_source_dir(
            title="Choose CSV file for all measurements",
            default_name=self._default_csv_name(),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if out_path is None:
            return None
        self.csv_path = out_path
        return self.csv_path

    def show_2d_window(self):
        self._set_active_button("show2d")
        return super().show_2d_window()

    def show_3d_window(self) -> None:
        self._set_active_button("show3d")
        path = self.last_integrated_3d_path
        if path is None or not path.exists():
            fin2d.messagebox.showinfo("3D output not ready", "Save a measurement first to generate the 3D wireframe.")
            return

        fin3d = self._load_fin3d()
        curves2d = getattr(self, "last_integrated_curves2d", None)
        fin_id = getattr(self, "last_integrated_fin_id", None) or path.stem
        if curves2d is None:
            if self.controls is None:
                fin2d.messagebox.showinfo("3D output not ready", "Save a measurement first to open the rotatable 3D view.")
                return
            curves2d = self._measured_curves_for_3d(fin3d, n_per_seg=234)

        try:
            fin3d.ensure_matplotlib()
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except Exception as exc:
            fin2d.messagebox.showwarning(
                "Interactive 3D unavailable",
                f"The 3D PNG was saved, but the rotatable Matplotlib/Tk view could not start:\n{type(exc).__name__}: {exc}",
            )
            return

        win = fin2d.tk.Toplevel(self)
        win.title(f"Rotatable 3D Wireframe - {path.name}")
        win.geometry("1100x900")

        top = fin2d.tk.Frame(win, bd=1, relief="raised")
        top.pack(side="top", fill="x")
        fin2d.tk.Button(
            top, text="Save current 3D view...",
            command=lambda: self.save_current_3d_view_dialog(fig, path.name, source_directory=export_snapshot["source_directory"]),
        ).pack(side="right", padx=8, pady=4)
        export_snapshot = self.__dict__.get("last_integrated_surface_snapshot")
        if export_snapshot is None:
            export_snapshot = self._surface_export_snapshot(curves2d, fin_id)
        fin2d.tk.Button(
            top, text="Export simulation surface...",
            command=lambda: self.export_surface_dialog(export_snapshot),
        ).pack(side="right", padx=8, pady=4)
        fin2d.tk.Label(top, text=str(path), anchor="w").pack(side="left", fill="x", expand=True, padx=8, pady=4)

        fig = Figure(figsize=(9.5, 7.0), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        controls = fin2d.tk.Frame(win, bd=1, relief="raised")
        controls.pack(side="bottom", fill="x")
        pitch_var = fin2d.tk.DoubleVar(value=18.0)
        yaw_var = fin2d.tk.DoubleVar(value=-65.0)
        roll_var = fin2d.tk.DoubleVar(value=0.0)

        def redraw(_=None):
            ax.clear()
            fin3d.draw_3d_wireframe_axes(
                ax, curves2d, fin_id, n_y_slices=39, n_chord_pts=117, n_z_slices=20,
                top_focus_power=3.0, edge_min_thickness=fin3d.DEFAULT_EDGE_MIN_THICKNESS,
                tip_min_chord=fin3d.DEFAULT_TIP_MIN_CHORD, pitch=pitch_var.get(),
                yaw=yaw_var.get(), roll=roll_var.get(),
            )
            canvas.draw_idle()

        self._add_3d_view_slider(controls, "Pitch", pitch_var, -90.0, 90.0, redraw)
        self._add_3d_view_slider(controls, "Yaw", yaw_var, -180.0, 180.0, redraw)
        self._add_3d_view_slider(controls, "Roll", roll_var, -180.0, 180.0, redraw)
        redraw()
        self.last_3d_view_figure = fig


    def _surface_export_snapshot(self, curves, fin_id):
        landmarks = {
            key: fin2d.standardize_contour_units(
                [point], self.landmarks["B"], self.landmarks["A"], self.landmarks["C30"], ab_units=10.0,
            )[0]
            for key, point in self.landmarks.items()
        }
        source = Path(self.image_path).expanduser().resolve()
        provenance = {
            "integrated_version": INTEGRATED_VERSION, "fin_id": fin_id,
            "geometry_kind": "individual_photo_fitted_reconstruction_not_canonical_centroid",
            "source_image": str(source),
            "source_image_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
            "measurement_csv": str(self.csv_path) if self.csv_path else None,
            "landmarks_image_coordinates": dict(self.landmarks),
            "landmark_placement_notes": copy.deepcopy(self.__dict__.get("landmark_placement_notes", [])),
            "bezier_controls_image_coordinates": asdict(self.controls),
            "crop_box": self.__dict__.get("crop_box"), "flip_x": self.__dict__.get("flip_x"), "rotate_deg": self.__dict__.get("rotate_deg"),
            "two_d_source_sha256": hashlib.sha256(FINSHAPE2D_PATH.read_bytes()).hexdigest(),
            "integrated_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        return copy.deepcopy({"curves": curves, "landmarks": landmarks, "provenance": provenance, "source_directory": source.parent})

    def _load_surface_exporter(self):
        if not SURFACE_EXPORT_PATH.is_file():
            fin2d.messagebox.showerror(
                "Surface exporter file missing",
                f"Restore the complete release package. Required file:\n{SURFACE_EXPORT_PATH}\n\n"
                "Your measurement and 2D/3D outputs are unchanged.",
                parent=self,
            )
            return None
        try:
            exporter = _load_module("finshape_cfd_0_1_0", SURFACE_EXPORT_PATH)
            if not callable(getattr(exporter.shapely, "constrained_delaunay_triangles", None)):
                raise ImportError("Surface export requires Shapely >=2.1 with constrained Delaunay triangulation.")
            if tuple(exporter.shapely.geos_version) < (3, 10, 0):
                raise ImportError("Surface export requires GEOS >=3.10.")
            return exporter
        except (ImportError, OSError) as exc:
            message = surface_setup_message(exc)
            print(f"[SURFACE SETUP] {message}")
            fin2d.messagebox.showerror("Surface export setup required", message, parent=self)
            return None

    def export_surface_dialog(self, snapshot):
        exporter = self._load_surface_exporter()
        if exporter is None:
            return
        print(f"[SURFACE] Export interpreter: {sys.executable}")
        win = fin2d.tk.Toplevel(self)
        win.title("Experimental simulation surface")
        win.resizable(False, False)
        fields = [
            ("Physical A-B length (metres)", ""),
            ("Scale source or assumed scenario", ""),
            ("Minimum full thickness (AB=10 units)", "0.05"),
        ]
        entries = []
        for row, (label, value) in enumerate(fields):
            fin2d.tk.Label(win, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=6)
            entry = fin2d.tk.Entry(win, width=40)
            entry.insert(0, value)
            entry.grid(row=row, column=1, padx=10, pady=6)
            entries.append(entry)
        fin2d.tk.Label(win, text="Triangle subdivisions").grid(row=3, column=0, sticky="w", padx=10)
        resolution = fin2d.tk.Spinbox(win, from_=1, to=3, width=5)
        resolution.delete(0, "end")
        resolution.insert(0, "2")
        resolution.grid(row=3, column=1, sticky="w", padx=10)
        fin2d.tk.Label(win, text=exporter.ASSUMPTION, wraplength=600, justify="left").grid(
            row=4, column=0, columnspan=2, padx=10, pady=12, sticky="w",
        )
        accepted = fin2d.tk.BooleanVar(value=False)
        fin2d.tk.Checkbutton(win, text="Accept experimental thickness scenario for this export", variable=accepted).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=10,
        )
        status = fin2d.tk.StringVar(value="Physical scale and modeling assumptions required.")
        fin2d.tk.Label(win, textvariable=status, wraplength=600, justify="left").grid(row=6, column=0, columnspan=2, padx=10, pady=8)

        def save():
            try:
                settings = exporter.SurfaceSettings(
                    ab_length_m=float(entries[0].get()), scale_provenance=entries[1].get(),
                    edge_min_thickness_units=float(entries[2].get()), subdivisions=int(resolution.get()),
                    accept_experimental_thickness=accepted.get(),
                )
                settings.validate()
                data = exporter.make_input(snapshot["curves"], snapshot["landmarks"], settings, snapshot["provenance"])
                target = self._ask_save_file_in_source_dir(
                    title="Save experimental simulation surface package",
                    default_name=f"FinShapeIntegrated_{INTEGRATED_VERSION}_surface_{sanitize_output_id(snapshot['provenance']['fin_id'])}.zip",
                    defaultextension=".zip", filetypes=[("Surface package", "*.zip")],
                    source_directory=snapshot["source_directory"],
                )
                if target is None:
                    return
                status.set("Constructing surface and checking STL/OBJ round trips...")
                win.update_idletasks()
                report = exporter.export_package(data, target, overwrite=True)
                status.set(f"Saved {report['triangles']} triangles: {target}\nExperimental surface; simulation setup remains incomplete.")
                self.status.set(f"Experimental surface package saved: {target}")
            except Exception as exc:
                status.set(f"Export failed: {exc}")
                fin2d.messagebox.showerror("Surface export failed", str(exc), parent=win)

        fin2d.tk.Button(win, text="Save surface package...", command=save).grid(row=7, column=1, sticky="e", padx=10, pady=10)
        entries[0].focus_set()

    def _add_3d_view_slider(self, parent, label: str, variable, from_: float, to: float, command) -> None:
        frame = fin2d.tk.Frame(parent)
        frame.pack(side="left", fill="x", expand=True, padx=8, pady=6)
        fin2d.tk.Label(frame, text=label).pack(side="left", padx=(0, 6))
        scale = fin2d.tk.Scale(
            frame, variable=variable, from_=from_, to=to, orient="horizontal",
            resolution=1.0, showvalue=True, command=command, length=280,
        )
        scale.pack(side="left", fill="x", expand=True)

    def save_current_3d_view_dialog(self, fig, default_name: str, source_directory=None) -> None:
        out_path = self._ask_save_file_in_source_dir(
            source_directory=source_directory,
            title="Save current 3D view as",
            default_name=default_name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if out_path is None:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=260, bbox_inches="tight")
        self.status.set(f"Current 3D view saved: {out_path}")

    def save_3d_image_dialog(self) -> None:
        path = self.last_integrated_3d_path
        if path is None or not path.exists():
            fin2d.messagebox.showinfo("3D output not ready", "Save a measurement first to generate the 3D wireframe.")
            return
        out_path = self._ask_save_file_in_source_dir(
            title="Save 3D wireframe image as",
            default_name=path.name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if out_path is None:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out_path)
        self.status.set(f"3D image saved: {out_path}")

    def save_contour_dialog(self, poly_units: List[Tuple[float, float]]):
        stem = self.image_path.stem if self.image_path else "contour"
        filetypes = [("SVG", "*.svg")]
        if fin2d.PIL_OK:
            filetypes += [("PNG", "*.png"), ("JPG", "*.jpg"), ("BMP", "*.bmp")]
        filetypes += [("PostScript", "*.ps")]

        out_path = self._ask_save_file_in_source_dir(
            title="Save contour image",
            default_name=f"{stem}_contour.svg",
            defaultextension=".svg",
            filetypes=filetypes,
        )
        if out_path is None:
            return
        ext = out_path.suffix.lower()

        if ext == ".svg":
            self._save_svg(poly_units, out_path)
            fin2d.messagebox.showinfo("Saved", f"Saved: {out_path.name}")
            return

        if ext in [".png", ".jpg", ".bmp"]:
            if not fin2d.PIL_OK:
                fin2d.messagebox.showwarning("Pillow missing", "Raster export requires Pillow (pip install pillow).")
                return
            self._save_raster_contour(poly_units, out_path, ext)
            fin2d.messagebox.showinfo("Saved", f"Saved: {out_path.name}")
            return

        self._save_ps_contour(poly_units, out_path)
        fin2d.messagebox.showinfo("Saved", f"Saved: {out_path.name}")

    def _load_fin3d(self):
        if self._fin3d_module is None:
            self._fin3d_module = _load_module("finshape3d_1_1_20", FINSHAPE3D_PATH)
        return self._fin3d_module

    def save_measurement(self):
        self._set_active_button("save")
        before = len(self.measurements)
        super().save_measurement()
        if len(self.measurements) <= before:
            return
        if not getattr(self, "locked_after_save", False):
            return
        try:
            self.generate_integrated_outputs(self.measurements[-1])
        except Exception as exc:
            print(f"[ERROR] Integrated 3D generation failed: {type(exc).__name__}: {exc}")
            fin2d.messagebox.showwarning(
                "Integrated 3D generation failed",
                f"The 2D measurement was saved, but 3D output failed:\n{type(exc).__name__}: {exc}",
            )
        finally:
            if getattr(self, "locked_after_save", False) and getattr(self, "btn_set_landmarks", None) is not None:
                self.btn_set_landmarks.configure(text="6. Set Landmarks")
            self._sync_active_button_to_mode()

    def generate_integrated_outputs(self, measurement_row: Dict[str, str]) -> List[Path]:
        fin3d = self._load_fin3d()
        outdir = output_dir_for_measurement(self.csv_path, self.image_path)
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[STARTUP] Integrated output directory: {outdir}")

        fallback_id = f"row{len(self.measurements) - 1}"
        row3d = measurement_row_to_3d_series(measurement_row, fin3d, fallback_id=fallback_id)
        fin_id = str(row3d["id"])
        warnings: List[str] = []

        generated: List[Path] = []
        measured_svg = outdir / f"FinShapeIntegrated_{INTEGRATED_VERSION}_2D_measured_{fin_id}.svg"
        self._save_integrated_measured_svg(measured_svg)
        generated.append(measured_svg)

        curves2d = self._measured_curves_for_3d(fin3d, n_per_seg=234)

        measured_2d = outdir / f"FinShapeIntegrated_{INTEGRATED_VERSION}_2D_measured_{fin_id}.png"
        wireframe_3d = outdir / f"FinShapeIntegrated_{INTEGRATED_VERSION}_3D_wireframe_{fin_id}.png"
        self._plot_measured_2d_reference(fin3d, curves2d, fin_id, measured_2d)
        edge_min_thickness = fin3d.DEFAULT_EDGE_MIN_THICKNESS
        tip_min_chord = fin3d.DEFAULT_TIP_MIN_CHORD
        warnings.append(
            "Structural visualization assumption applied: "
            f"minimum tip/trailing-edge thickness={edge_min_thickness:g} AB units and "
            f"minimum tip chord={tip_min_chord:g} AB units were imposed to prevent geometric collapse near the apex; "
            "these values are not measured biological thickness and do not validate CFD geometry."
        )
        fin3d.plot_3d_wireframe(
            curves2d, fin_id, wireframe_3d, n_y_slices=39, n_chord_pts=117, n_z_slices=20,
            top_focus_power=3.0, edge_min_thickness=edge_min_thickness, tip_min_chord=tip_min_chord,
        )
        generated.extend([measured_2d, wireframe_3d])
        self.last_integrated_3d_path = wireframe_3d
        self.last_integrated_curves2d = curves2d
        self.last_integrated_fin_id = fin_id
        self.last_integrated_surface_snapshot = self._surface_export_snapshot(curves2d, fin_id)
        self._set_show3d_enabled(True)

        manifest = outdir / f"FinShapeIntegrated_{INTEGRATED_VERSION}_manifest_{fin_id}.txt"
        manifest.write_text(self._build_manifest(fin_id, outdir, generated, warnings), encoding="utf-8")
        generated.append(manifest)

        for path in generated:
            print(f"[OK] integrated output: {path}")
        print(f"[DONE] Integrated output directory: {outdir}")
        self.status.set(f"Integrated 2D/3D outputs saved: {outdir}")
        return generated

    def _measured_curves_for_3d(self, fin3d, n_per_seg: int = 234):
        if self.controls is None:
            raise ValueError("No fitted 2D controls are available for 3D rendering.")
        for key in ("B", "A", "C30"):
            if key not in self.landmarks:
                raise ValueError(f"Missing {key} landmark for measured 3D rendering.")
        raw = fin2d.curves_from_controls(self.controls, n_per_seg=n_per_seg)

        def std(points):
            return fin3d.np.array(
                fin2d.standardize_contour_units(
                    points,
                    self.landmarks["B"],
                    self.landmarks["A"],
                    self.landmarks["C30"],
                    ab_units=10.0,
                ),
                dtype=float,
            )

        le0 = std(raw["LE0"])
        le1 = std(raw["LE1"])
        te1 = std(raw["TE1"])
        te2 = std(raw["TE2"])
        te3 = std(raw["TE3"])
        leading = fin3d.np.vstack([le0, le1[1:]])
        trailing = fin3d.np.vstack([te1, te2[1:], te3[1:]])
        outline = fin3d.np.vstack([leading, trailing[1:]])
        return {
            "leading": leading,
            "le0": le0,
            "le1": le1,
            "te1": te1,
            "te2": te2,
            "te3": te3,
            "trailing": trailing,
            "outline": outline,
        }

    def _plot_measured_2d_reference(self, fin3d, curves2d, fin_id: str, out_png: Path) -> None:
        fin3d.ensure_matplotlib()
        outline = curves2d["outline"]
        fig, ax = fin3d.plt.subplots(figsize=(10, 8))
        ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.8)
        ax.scatter([0.0], [0.0], s=30, color="black")
        ax.axhline(0.0, color="0.70", linewidth=0.8)
        ax.axvline(0.0, color="0.70", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (AB=10 units)")
        ax.set_ylabel("Y (AB=10 units)")
        ax.set_title(f"{fin_id}: fitted measured 2D contour (FinShape Integrated {INTEGRATED_VERSION})")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=260, bbox_inches="tight")
        fin3d.plt.close(fig)

    def _save_integrated_measured_svg(self, out_path: Path) -> None:
        if self.controls is None:
            raise ValueError("No fitted 2D controls are available.")
        required = ("B", "A", "C30")
        if any(k not in self.landmarks for k in required):
            raise ValueError("Missing B, A, or C30 landmark for measured 2D contour export.")
        poly = fin2d.outline_polygon(self.controls, n_per_seg=1100)
        poly_units = fin2d.standardize_contour_units(
            poly,
            self.landmarks["B"],
            self.landmarks["A"],
            self.landmarks["C30"],
            ab_units=10.0,
        )
        self._save_svg(poly_units, out_path)

    def _build_manifest(self, fin_id: str, outdir: Path, outputs: List[Path], warnings: List[str]) -> str:
        lines = [
            f"FinShape Surfer GUI version: {INTEGRATED_VERSION}",
            f"Operator-accepted off-guide clicks: {self.__dict__.get('landmark_placement_notes', [])!r}",
            f"Fin ID: {fin_id}",
            f"2D source: {FINSHAPE2D_PATH}",
            f"3D source: {FINSHAPE3D_PATH}",
            f"Input image: {self.image_path}",
            f"Measurement CSV: {self.csv_path}",
            f"Output directory: {outdir}",
            "3D source outline: fitted measured 2D Bezier contour, standardized to AB=10 units.",
            "3D density: measured curve segments n=234 each, y_slices=39, chord_points=117, contour_reference_points=100, chordwise_rails<=35, z_slices=20",
            "3D contour anchors: measured Bezier segment endpoints are preserved for edge-reference sampling.",
            "3D wireframe rails: chordwise surface rails span from leading edge to trailing edge.",
            "3D tip rendering: readable chordwise rails converge to the measured apex using a documented structural visualization floor.",
            "Structural visualization assumption: minimum tip/trailing-edge thickness and minimum tip chord are imposed to prevent geometric collapse near the apex; this is not measured biological thickness and not CFD validation.",
            "3D thickness model: FinShape3D scientific section/thickness equations are unchanged; only visualization drawing applies the structural floor.",
            "Outputs:",
        ]
        lines.extend(f"- {path}" for path in outputs)
        lines.append("Warnings:")
        if warnings:
            lines.extend(f"- {warning}" for warning in warnings)
        else:
            lines.append("- none")
        lines.append("Scope: integrated 2D measurement and 3D visualization of the fitted outline; not CFD-ready mesh generation.")
        return "\n".join(lines) + "\n"


def main() -> None:
    app = IntegratedFinShapeApp()
    if not fin2d.PIL_OK:
        fin2d.messagebox.showwarning(
            "Pillow optional",
            "Pillow is not installed. JPEG/flip/rotate/crop/zoom/raster exports will not work.\nInstall: pip install pillow",
        )
    app.mainloop()


if __name__ == "__main__":
    main()




