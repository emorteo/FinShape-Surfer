#!/usr/bin/env python3
"""FinShape Surfer 1.2.0: experimental one-side photograph draping.

The tested 1.1.3 workflow remains the implementation for image measurement,
2D output, wireframe output and experimental surface export. This version adds
an isolated, reviewable texture-registration layer to the rotatable 3D window.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Optional


INTEGRATED_VERSION = "1.2.0"
CODE_DIR = Path(__file__).resolve().parent
BASE_GUI_PATH = CODE_DIR / "FinShape_Surfer_GUI_1.1.3.py"
TEXTURE_RENDERER_PATH = CODE_DIR / "FinShapeTexture_0.1.0.py"


def _load_module(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = _load_module("finshape_surfer_gui_1_1_3_base", BASE_GUI_PATH)
# Base methods resolve this global at runtime. Changing it in the loaded module
# versions output names while leaving the 1.1.3 source file untouched.
base.INTEGRATED_VERSION = INTEGRATED_VERSION
fin2d = base.fin2d


class IntegratedFinShapeApp(base.IntegratedFinShapeApp):
    def __init__(self):
        super().__init__()
        self._texture_module = None
        self.last_integrated_texture_snapshot = None
        self.last_texture_metadata_path: Optional[Path] = None
        self._append_texture_instructions()

    def _append_texture_instructions(self) -> None:
        if getattr(self, "info", None) is None:
            return
        self.info.configure(state="normal")
        self.info.insert(
            "end",
            "\nPhoto drape (experimental):\n"
            "  In Show 3D, choose Left or Right as the photographed anatomical side.\n"
            "  Image flip is recorded separately and does not change anatomical side.\n"
            "  The overlay has no camera-pose or perspective correction and is for visual inspection only.\n",
        )
        self.info.configure(state="disabled")

    def _clear_integrated_output_state(self):
        super()._clear_integrated_output_state()
        self.last_integrated_texture_snapshot = None
        self.last_texture_metadata_path = None

    def _surface_export_snapshot(self, curves, fin_id):
        snapshot = super()._surface_export_snapshot(curves, fin_id)
        provenance = snapshot["provenance"]
        provenance["integrated_base_source"] = str(BASE_GUI_PATH)
        provenance["integrated_base_source_sha256"] = hashlib.sha256(BASE_GUI_PATH.read_bytes()).hexdigest()
        provenance["integrated_source"] = str(Path(__file__).resolve())
        provenance["integrated_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        provenance["texture_renderer"] = str(TEXTURE_RENDERER_PATH)
        provenance["texture_renderer_sha256"] = (
            hashlib.sha256(TEXTURE_RENDERER_PATH.read_bytes()).hexdigest()
            if TEXTURE_RENDERER_PATH.is_file() else None
        )
        return snapshot

    def _texture_registration_snapshot(self, curves, fin_id, surface_snapshot, n_per_seg: int = 234):
        if not fin2d.PIL_OK or getattr(self, "pil_image", None) is None:
            return None
        if self.controls is None:
            raise ValueError("No fitted Bezier controls are available for photograph draping.")
        raw = fin2d.curves_from_controls(self.controls, n_per_seg=n_per_seg)
        image_leading = list(raw["LE0"]) + list(raw["LE1"])[1:]
        image_trailing = list(raw["TE1"]) + list(raw["TE2"])[1:] + list(raw["TE3"])[1:]
        image_outline = image_leading + image_trailing[1:]
        if len(image_outline) != len(curves["outline"]):
            raise ValueError("Image and standardized Bezier outlines have different sample counts.")

        working_image = self._apply_transforms().convert("RGB").copy()
        source = Path(self.image_path).expanduser().resolve()
        return {
            "schema": "finshape.texture-snapshot/1",
            "fin_id": fin_id,
            "working_image": working_image,
            "working_image_sha256": hashlib.sha256(working_image.tobytes()).hexdigest(),
            "working_image_size_px": list(working_image.size),
            "source_image": str(source),
            "source_image_sha256": surface_snapshot["provenance"].get("source_image_sha256"),
            "source_directory": source.parent,
            "curves": copy.deepcopy(curves),
            "image_outline": copy.deepcopy(image_outline),
            "image_landmarks": copy.deepcopy(self.landmarks),
            "standardized_landmarks": copy.deepcopy(surface_snapshot["landmarks"]),
            "bezier_controls_image_coordinates": copy.deepcopy(surface_snapshot["provenance"].get("bezier_controls_image_coordinates")),
            "flip_applied": bool(self.flip_x),
            "rotate_deg": float(self.rotate_deg),
            "crop_box": copy.deepcopy(self.crop_box),
        }

    def generate_integrated_outputs(self, measurement_row):
        outputs = super().generate_integrated_outputs(measurement_row)
        try:
            snapshot = self.last_integrated_surface_snapshot
            self.last_integrated_texture_snapshot = self._texture_registration_snapshot(
                self.last_integrated_curves2d,
                self.last_integrated_fin_id,
                snapshot,
            )
        except Exception as exc:
            self.last_integrated_texture_snapshot = None
            print(f"[WARN] Photograph drape snapshot unavailable: {type(exc).__name__}: {exc}")
        return outputs

    def _build_manifest(self, fin_id, outdir, outputs, warnings):
        inherited = super()._build_manifest(fin_id, outdir, outputs, warnings)
        addition = (
            f"Photo drape snapshot: transformed measurement image captured; flip_applied={bool(self.flip_x)}.\n"
            "Photo-side rule: horizontal image flipping changes pixel orientation, not anatomical side; "
            "Left or Right must be selected explicitly in Show 3D.\n"
            "Photo drape method: inverse AB=10 planform transform with exact B-A-C30 anchors, complete fitted "
            "Bezier-boundary verification, and untextured B-C30 base attachment.\n"
            "Photo drape status: experimental visual registration only; no camera calibration, lens correction, "
            "camera pose, visibility solution, perspective correction or identity decision.\n"
        )
        return inherited.replace("Scope:", addition + "Scope:")

    def _load_texture_renderer(self):
        if self._texture_module is not None:
            return self._texture_module
        if not TEXTURE_RENDERER_PATH.is_file():
            fin2d.messagebox.showerror(
                "Texture renderer file missing",
                f"Restore the complete 1.2.0 release package. Required file:\n{TEXTURE_RENDERER_PATH}",
                parent=self,
            )
            return None
        try:
            self._texture_module = _load_module("finshape_texture_0_1_0", TEXTURE_RENDERER_PATH)
        except (ImportError, OSError, ValueError) as exc:
            fin2d.messagebox.showerror(
                "Texture renderer unavailable",
                f"The wireframe remains available. Texture setup failed:\n{type(exc).__name__}: {exc}",
                parent=self,
            )
            return None
        return self._texture_module

    def _write_texture_metadata(self, renderer, scene, snapshot) -> Path:
        metadata = renderer.scene_metadata(scene, snapshot)
        metadata["surfer_version"] = INTEGRATED_VERSION
        metadata["surfer_source"] = str(Path(__file__).resolve())
        metadata["surfer_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        outdir = self.last_integrated_3d_path.parent
        target = outdir / (
            f"FinShapeSurfer_{INTEGRATED_VERSION}_texture_registration_"
            f"{scene.photo_side}_{base.sanitize_output_id(snapshot['fin_id'])}.json"
        )
        target.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        self.last_texture_metadata_path = target
        print(f"[OK] texture registration metadata: {target}")
        return target

    def show_3d_window(self) -> None:
        self._set_active_button("show3d")
        path = self.last_integrated_3d_path
        if path is None or not path.exists():
            fin2d.messagebox.showinfo("3D output not ready", "Save a measurement first to generate the 3D wireframe.")
            return

        fin3d = self._load_fin3d()
        curves2d = self.last_integrated_curves2d
        fin_id = self.last_integrated_fin_id or path.stem
        if curves2d is None:
            fin2d.messagebox.showinfo("3D output not ready", "Save a measurement first to open the rotatable 3D view.")
            return
        try:
            fin3d.ensure_matplotlib()
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except Exception as exc:
            fin2d.messagebox.showwarning(
                "Interactive 3D unavailable",
                f"The 3D PNG was saved, but the rotatable view could not start:\n{type(exc).__name__}: {exc}",
            )
            return

        export_snapshot = self.last_integrated_surface_snapshot or self._surface_export_snapshot(curves2d, fin_id)
        texture_snapshot = self.last_integrated_texture_snapshot
        win = fin2d.tk.Toplevel(self)
        win.title(f"Rotatable 3D Inspection - {path.name}")
        win.geometry("1180x920")

        top = fin2d.tk.Frame(win, bd=1, relief="raised")
        top.pack(side="top", fill="x")
        fin2d.tk.Label(top, text=str(path), anchor="w").pack(side="left", fill="x", expand=True, padx=8, pady=4)

        mode_bar = fin2d.tk.Frame(win, bd=1, relief="groove")
        mode_bar.pack(side="top", fill="x")
        mode_var = fin2d.tk.StringVar(value="wireframe")
        side_var = fin2d.tk.StringVar(value="")
        status_var = fin2d.tk.StringVar(value="Wireframe view. Select the photographed side before enabling the photo drape.")
        pitch_var = fin2d.tk.DoubleVar(value=18.0)
        yaw_var = fin2d.tk.DoubleVar(value=-65.0)
        roll_var = fin2d.tk.DoubleVar(value=0.0)

        fig = Figure(figsize=(9.8, 7.0), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        canvas = FigureCanvasTkAgg(fig, master=win)

        controls = fin2d.tk.Frame(win, bd=1, relief="raised")
        controls.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        scenes = {}
        pending_view = {"after": None}

        def apply_view():
            pending_view["after"] = None
            try:
                ax.view_init(elev=float(pitch_var.get()), azim=float(yaw_var.get()), roll=float(roll_var.get()))
            except TypeError:
                ax.view_init(elev=float(pitch_var.get()), azim=float(yaw_var.get()))
            canvas.draw_idle()

        def schedule_view(_=None):
            if pending_view["after"] is not None:
                win.after_cancel(pending_view["after"])
            pending_view["after"] = win.after(45, apply_view)

        def build_scene(side):
            if side in scenes:
                return scenes[side]
            if texture_snapshot is None:
                raise RuntimeError("No transformed measurement-image snapshot is available. Save the measurement again with Pillow enabled.")
            renderer = self._load_texture_renderer()
            if renderer is None:
                raise RuntimeError("Texture renderer is unavailable.")
            exporter = self._load_surface_exporter()
            if exporter is None:
                raise RuntimeError("The optional surface packages required by the texture mesh are unavailable.")
            scene = renderer.build_texture_scene(
                texture_snapshot,
                exporter,
                side,
                edge_min_thickness=fin3d.DEFAULT_EDGE_MIN_THICKNESS,
                subdivisions=0,
            )
            scenes[side] = (renderer, scene)
            self._write_texture_metadata(renderer, scene, texture_snapshot)
            return scenes[side]

        def render_full():
            if mode_var.get() == "photo":
                side = side_var.get()
                if not side:
                    mode_var.set("wireframe")
                    fin2d.messagebox.showinfo(
                        "Photographed side required",
                        "Select Left or Right. Image flipping is tracked separately and does not determine anatomical side.",
                        parent=win,
                    )
                else:
                    try:
                        renderer, scene = build_scene(side)
                        renderer.draw_texture_scene(
                            ax, scene, fin3d, fin_id,
                            pitch=pitch_var.get(), yaw=yaw_var.get(), roll=roll_var.get(),
                        )
                        status_var.set(
                            f"Experimental {side}-side photo drape | transformed image flipped: "
                            f"{'yes' if texture_snapshot['flip_applied'] else 'no'} | "
                            f"texture coverage: {scene.coverage_fraction:.1%} | no camera-pose correction"
                        )
                        canvas.draw_idle()
                        return
                    except Exception as exc:
                        mode_var.set("wireframe")
                        fin2d.messagebox.showerror(
                            "Photograph drape failed",
                            f"The wireframe remains available and no measurement data changed.\n\n{type(exc).__name__}: {exc}",
                            parent=win,
                        )
            ax.clear()
            fin3d.draw_3d_wireframe_axes(
                ax, curves2d, fin_id, n_y_slices=39, n_chord_pts=117, n_z_slices=20,
                top_focus_power=3.0, edge_min_thickness=fin3d.DEFAULT_EDGE_MIN_THICKNESS,
                tip_min_chord=fin3d.DEFAULT_TIP_MIN_CHORD, pitch=pitch_var.get(),
                yaw=yaw_var.get(), roll=roll_var.get(),
            )
            status_var.set("Wireframe view. The measured contour, landmarks and thickness equations are unchanged.")
            canvas.draw_idle()

        def select_side():
            side = side_var.get()
            yaw_var.set(65.0 if side == "left" else -65.0)
            if mode_var.get() == "photo":
                render_full()
            else:
                apply_view()

        def save_current_view():
            if mode_var.get() == "photo" and side_var.get():
                filename = (
                    f"FinShapeSurfer_{INTEGRATED_VERSION}_3D_photo_drape_"
                    f"{side_var.get()}_{base.sanitize_output_id(fin_id)}.png"
                )
            else:
                filename = path.name
            self.save_current_3d_view_dialog(
                fig, filename, source_directory=export_snapshot["source_directory"],
            )

        fin2d.tk.Button(top, text="Save current 3D view...", command=save_current_view).pack(side="right", padx=8, pady=4)
        fin2d.tk.Button(
            top, text="Export simulation surface...",
            command=lambda: self.export_surface_dialog(export_snapshot),
        ).pack(side="right", padx=8, pady=4)

        fin2d.tk.Label(mode_bar, text="View:").pack(side="left", padx=(8, 3), pady=4)
        fin2d.tk.Radiobutton(
            mode_bar, text="Wireframe", variable=mode_var, value="wireframe",
            indicatoron=False, command=render_full,
        ).pack(side="left", padx=2, pady=4)
        fin2d.tk.Radiobutton(
            mode_bar, text="Measured photo", variable=mode_var, value="photo",
            indicatoron=False, command=render_full,
        ).pack(side="left", padx=2, pady=4)
        fin2d.tk.Label(mode_bar, text="Photographed side:").pack(side="left", padx=(16, 3), pady=4)
        fin2d.tk.Radiobutton(
            mode_bar, text="Left (+Z)", variable=side_var, value="left",
            indicatoron=False, command=select_side,
        ).pack(side="left", padx=2, pady=4)
        fin2d.tk.Radiobutton(
            mode_bar, text="Right (-Z)", variable=side_var, value="right",
            indicatoron=False, command=select_side,
        ).pack(side="left", padx=2, pady=4)
        flip_text = "yes" if texture_snapshot is not None and texture_snapshot["flip_applied"] else "no"
        fin2d.tk.Label(mode_bar, text=f"Image flipped for measurement: {flip_text}").pack(side="left", padx=16, pady=4)
        fin2d.tk.Label(win, textvariable=status_var, anchor="w").pack(side="bottom", fill="x", padx=8, pady=3)

        self._add_3d_view_slider(controls, "Pitch", pitch_var, -90.0, 90.0, schedule_view)
        self._add_3d_view_slider(controls, "Yaw", yaw_var, -180.0, 180.0, schedule_view)
        self._add_3d_view_slider(controls, "Roll", roll_var, -180.0, 180.0, schedule_view)
        render_full()
        self.last_3d_view_figure = fig


def main() -> None:
    app = IntegratedFinShapeApp()
    if not fin2d.PIL_OK:
        fin2d.messagebox.showwarning(
            "Pillow required for photo drape",
            "Pillow is not installed. Measurement formats and photograph draping are limited.\nInstall: pip install pillow",
        )
    app.mainloop()


if __name__ == "__main__":
    main()
