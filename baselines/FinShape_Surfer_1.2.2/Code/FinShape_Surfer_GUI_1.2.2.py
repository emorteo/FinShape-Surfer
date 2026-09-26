#!/usr/bin/env python3
"""FinShape Surfer 1.2.2: corrected, smooth high-resolution Photo-ID view.

The tested 1.2.1 application remains authoritative for measurement, 2D output,
wireframe construction and experimental CFD export. This additive version only
changes the high-resolution display and its 3D-window controls.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


INTEGRATED_VERSION = "1.2.2"
CODE_DIR = Path(__file__).resolve().parent
BASE_GUI_PATH = CODE_DIR / "FinShape_Surfer_GUI_1.2.1.py"
HQ_RENDERER_PATH = CODE_DIR / "FinShapeTextureHQ_0.2.0.py"
PHOTO_DISPLAY_SUBDIVISIONS = 0
PHOTO_INTERACTIVE_SUPERSAMPLE = 2
PHOTO_SAVE_SIZE = (1920, 1440)
PHOTO_SAVE_SUPERSAMPLE = 2


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


base = _load_module("finshape_surfer_gui_1_2_1_base", BASE_GUI_PATH)
base.INTEGRATED_VERSION = INTEGRATED_VERSION
base.base.INTEGRATED_VERSION = INTEGRATED_VERSION
base.base.base.INTEGRATED_VERSION = INTEGRATED_VERSION
fin2d = base.fin2d


class IntegratedFinShapeApp(base.IntegratedFinShapeApp):
    def _surface_export_snapshot(self, curves, fin_id):
        snapshot = super()._surface_export_snapshot(curves, fin_id)
        provenance = snapshot["provenance"]
        source = Path(__file__).resolve()
        provenance["hq_display_extension_source"] = str(source)
        provenance["hq_display_extension_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        provenance["hq_display_renderer"] = str(HQ_RENDERER_PATH)
        provenance["hq_display_renderer_sha256"] = (
            hashlib.sha256(HQ_RENDERER_PATH.read_bytes()).hexdigest()
            if HQ_RENDERER_PATH.is_file() else None
        )
        return snapshot

    def _build_manifest(self, fin_id, outdir, outputs, warnings):
        inherited = super()._build_manifest(fin_id, outdir, outputs, warnings)
        addition = (
            "Photo-ID display 1.2.2: corrected Left/Right skin mapping; Matplotlib-compatible camera angles, "
            "back-face culling and 2x silhouette antialiasing; edge-wall faces hidden only in the "
            "high-resolution photo view. Measurement and CFD geometry are unchanged.\n"
        )
        return inherited.replace("Scope:", addition + "Scope:")

    def _load_hq_renderer(self):
        if self._hq_texture_module is not None and getattr(self._hq_texture_module, "VERSION", None) == "0.2.0":
            return self._hq_texture_module
        if not HQ_RENDERER_PATH.is_file():
            fin2d.messagebox.showerror(
                "High-resolution renderer missing",
                f"Restore the complete {INTEGRATED_VERSION} release package. Required file:\n{HQ_RENDERER_PATH}",
                parent=self,
            )
            return None
        try:
            self._hq_texture_module = _load_module("finshape_texture_hq_0_2_0", HQ_RENDERER_PATH)
        except (ImportError, OSError, ValueError) as exc:
            fin2d.messagebox.showerror(
                "High-resolution renderer unavailable",
                f"Wireframe view remains available.\n\n{type(exc).__name__}: {exc}",
                parent=self,
            )
            return None
        return self._hq_texture_module

    def _write_hq_metadata(
        self, hq_renderer, hq_scene, render_result, base_renderer, texture_snapshot,
        *, pitch=None, yaw=None, roll=None,
    ) -> Path:
        metadata = base_renderer.scene_metadata(hq_scene.base_scene, texture_snapshot)
        metadata.update(hq_renderer.renderer_metadata(hq_scene, render_result))
        metadata["surfer_version"] = INTEGRATED_VERSION
        metadata["surfer_source"] = str(Path(__file__).resolve())
        metadata["surfer_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        metadata["geometric_skin"] = hq_scene.textured_skin
        metadata["display_triangle_subdivisions"] = PHOTO_DISPLAY_SUBDIVISIONS
        metadata["display_refinement_only"] = False
        metadata["measurement_geometry_changed"] = False
        metadata["cfd_surface_changed"] = False
        if pitch is not None and yaw is not None and roll is not None:
            metadata["render_view_degrees"] = {
                "pitch": float(pitch), "yaw": float(yaw), "roll": float(roll),
            }
        output_directory = self.last_integrated_3d_path.parent
        target = output_directory / (
            f"FinShapeSurfer_{INTEGRATED_VERSION}_hq_texture_registration_"
            f"{hq_scene.photo_side}_{base.base.base.sanitize_output_id(texture_snapshot['fin_id'])}.json"
        )
        target.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        self.last_hq_metadata_path = target
        print(f"[OK] high-resolution texture metadata: {target}")
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
        texture_snapshot = self.last_integrated_texture_snapshot
        if curves2d is None:
            fin2d.messagebox.showinfo("3D output not ready", "Save a measurement first to open the rotatable 3D view.")
            return
        try:
            fin3d.ensure_matplotlib()
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            from PIL import ImageTk
        except Exception as exc:
            fin2d.messagebox.showwarning(
                "Interactive 3D unavailable",
                f"The 3D PNG was saved, but the inspection window could not start:\n{type(exc).__name__}: {exc}",
            )
            return

        export_snapshot = self.last_integrated_surface_snapshot or self._surface_export_snapshot(curves2d, fin_id)
        win = fin2d.tk.Toplevel(self)
        win.title(f"Rotatable 3D Photo-ID Inspection - {path.name}")
        win.geometry("1180x920")

        top = fin2d.tk.Frame(win, bd=1, relief="raised")
        top.pack(side="top", fill="x")
        path_label = fin2d.tk.Label(top, text=str(path), anchor="w")

        mode_bar = fin2d.tk.Frame(win, bd=1, relief="groove")
        mode_bar.pack(side="top", fill="x")
        mode_var = fin2d.tk.StringVar(value="wireframe")
        side_var = fin2d.tk.StringVar(value="")
        status_var = fin2d.tk.StringVar(value="Wireframe view. Select the photographed side before photo mode.")
        save_button_text = fin2d.tk.StringVar(value="Save wireframe view...")
        pitch_var = fin2d.tk.DoubleVar(value=18.0)
        yaw_var = fin2d.tk.DoubleVar(value=-65.0)
        roll_var = fin2d.tk.DoubleVar(value=0.0)

        display = fin2d.tk.Frame(win, background="#f8f9fa")
        display.pack(side="top", fill="both", expand=True)
        figure = Figure(figsize=(9.8, 7.0), dpi=100)
        axis = figure.add_subplot(111, projection="3d")
        mpl_canvas = FigureCanvasTkAgg(figure, master=display)
        mpl_widget = mpl_canvas.get_tk_widget()
        photo_label = fin2d.tk.Label(display, background="#f8f9fa", bd=0)

        controls = fin2d.tk.Frame(win, bd=1, relief="raised")
        controls.pack(side="bottom", fill="x")
        fin2d.tk.Label(win, textvariable=status_var, anchor="w").pack(side="bottom", fill="x", padx=8, pady=3)

        scenes = {}
        metadata_written = set()
        pending = {"after": None, "resize_after": None}

        def current_photo_dimensions():
            win.update_idletasks()
            width = max(320, min(1400, display.winfo_width() - 4))
            height = max(240, min(1000, display.winfo_height() - 4))
            return width, height

        def show_matplotlib():
            photo_label.pack_forget()
            if not mpl_widget.winfo_ismapped():
                mpl_widget.pack(side="top", fill="both", expand=True)

        def show_photo():
            mpl_widget.pack_forget()
            if not photo_label.winfo_ismapped():
                photo_label.pack(side="top", fill="both", expand=True)

        def build_hq_scene(side):
            if side in scenes:
                return scenes[side]
            if scenes:
                base_renderer, hq_renderer, seed_scene = scenes[next(iter(scenes))]
                hq_scene = hq_renderer.retarget_photo_side(seed_scene, side)
                scenes[side] = (base_renderer, hq_renderer, hq_scene)
                return scenes[side]
            if texture_snapshot is None:
                raise RuntimeError("No transformed measurement-image snapshot is available. Save the measurement again.")
            base_renderer = self._load_texture_renderer()
            hq_renderer = self._load_hq_renderer()
            exporter = self._load_surface_exporter()
            if base_renderer is None or hq_renderer is None or exporter is None:
                raise RuntimeError("One or more texture-rendering components are unavailable.")
            base_scene = base_renderer.build_texture_scene(
                texture_snapshot,
                exporter,
                side,
                edge_min_thickness=fin3d.DEFAULT_EDGE_MIN_THICKNESS,
                subdivisions=PHOTO_DISPLAY_SUBDIVISIONS,
            )
            hq_scene = hq_renderer.build_hq_scene(base_scene, texture_snapshot, base_renderer)
            scenes[side] = (base_renderer, hq_renderer, hq_scene)
            return scenes[side]

        def render_photo():
            side = side_var.get()
            if not side:
                mode_var.set("wireframe")
                save_button_text.set("Save wireframe view...")
                fin2d.messagebox.showinfo(
                    "Photographed side required",
                    "Select Left or Right. Image flipping is recorded separately and does not determine anatomical side.",
                    parent=win,
                )
                render_wireframe()
                return
            try:
                base_renderer, hq_renderer, hq_scene = build_hq_scene(side)
                width, height = current_photo_dimensions()
                result = hq_renderer.render_hq_frame(
                    hq_scene,
                    width,
                    height,
                    pitch=pitch_var.get(),
                    yaw=yaw_var.get(),
                    roll=roll_var.get(),
                    supersample=PHOTO_INTERACTIVE_SUPERSAMPLE,
                )
                tk_image = ImageTk.PhotoImage(result.image)
                photo_label.configure(image=tk_image)
                photo_label.image = tk_image
                show_photo()
                self.last_hq_scene = hq_scene
                self.last_hq_render_result = result
                self.last_hq_view_image = result.image.copy()
                if side not in metadata_written:
                    self._write_hq_metadata(
                        hq_renderer, hq_scene, result, base_renderer, texture_snapshot,
                        pitch=pitch_var.get(), yaw=yaw_var.get(), roll=roll_var.get(),
                    )
                    metadata_written.add(side)
                status_var.set(
                    f"High-resolution {side}-side photo | {result.image.width}x{result.image.height} | "
                    f"2x antialias | {result.textured_pixels:,} internal photo samples | "
                    f"{len(hq_scene.faces):,} display triangles | {result.render_seconds:.2f} s | "
                    "edge wall hidden; CFD unchanged"
                )
            except Exception as exc:
                mode_var.set("wireframe")
                save_button_text.set("Save wireframe view...")
                render_wireframe()
                fin2d.messagebox.showerror(
                    "High-resolution photo rendering failed",
                    f"Wireframe view remains available and no measurement data changed.\n\n{type(exc).__name__}: {exc}",
                    parent=win,
                )

        def render_wireframe():
            show_matplotlib()
            axis.clear()
            fin3d.draw_3d_wireframe_axes(
                axis, curves2d, fin_id, n_y_slices=39, n_chord_pts=117, n_z_slices=20,
                top_focus_power=3.0, edge_min_thickness=fin3d.DEFAULT_EDGE_MIN_THICKNESS,
                tip_min_chord=fin3d.DEFAULT_TIP_MIN_CHORD, pitch=pitch_var.get(),
                yaw=yaw_var.get(), roll=roll_var.get(),
            )
            axis.set_proj_type("ortho")
            mpl_canvas.draw_idle()
            status_var.set("Wireframe view. Measured contour, thickness equations and CFD geometry are unchanged.")

        def render_current():
            pending["after"] = None
            if mode_var.get() == "photo":
                render_photo()
            else:
                try:
                    axis.view_init(elev=pitch_var.get(), azim=yaw_var.get(), roll=roll_var.get())
                except TypeError:
                    axis.view_init(elev=pitch_var.get(), azim=yaw_var.get())
                mpl_canvas.draw_idle()

        def schedule_render(_=None):
            if pending["after"] is not None:
                win.after_cancel(pending["after"])
            pending["after"] = win.after(140 if mode_var.get() == "photo" else 45, render_current)

        def on_resize(_event=None):
            if mode_var.get() != "photo":
                return
            if pending["resize_after"] is not None:
                win.after_cancel(pending["resize_after"])
            pending["resize_after"] = win.after(220, render_photo)

        def change_mode():
            if mode_var.get() == "photo":
                save_button_text.set("Save high-resolution photo...")
                render_photo()
            else:
                save_button_text.set("Save wireframe view...")
                render_wireframe()

        def select_side():
            # Anatomical side selection must never overwrite the current camera.
            if mode_var.get() == "photo":
                render_photo()

        def save_current_view():
            if mode_var.get() != "photo" or not side_var.get():
                self.save_current_3d_view_dialog(
                    figure, path.name, source_directory=export_snapshot["source_directory"],
                )
                return
            side = side_var.get()
            output = self._ask_save_file_in_source_dir(
                source_directory=export_snapshot["source_directory"],
                title="Save high-resolution 3D Photo-ID view as",
                default_name=(
                    f"FinShapeSurfer_{INTEGRATED_VERSION}_3D_photo_drape_HQ_"
                    f"{side}_{base.base.base.sanitize_output_id(fin_id)}.png"
                ),
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            )
            if output is None:
                return
            try:
                base_renderer, hq_renderer, hq_scene = build_hq_scene(side)
                status_var.set("Rendering 1920 x 1440 supersampled Photo-ID image...")
                win.update_idletasks()
                result = hq_renderer.render_hq_frame(
                    hq_scene,
                    PHOTO_SAVE_SIZE[0],
                    PHOTO_SAVE_SIZE[1],
                    pitch=pitch_var.get(),
                    yaw=yaw_var.get(),
                    roll=roll_var.get(),
                    supersample=PHOTO_SAVE_SUPERSAMPLE,
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                result.image.save(output, format="PNG", optimize=False)
                self._write_hq_metadata(
                    hq_renderer, hq_scene, result, base_renderer, texture_snapshot,
                    pitch=pitch_var.get(), yaw=yaw_var.get(), roll=roll_var.get(),
                )
                self.status.set(f"High-resolution 3D Photo-ID view saved: {output}")
                status_var.set(
                    f"Saved {result.image.width}x{result.image.height}: {output} | render {result.render_seconds:.2f} s"
                )
            except Exception as exc:
                fin2d.messagebox.showerror("High-resolution save failed", str(exc), parent=win)

        save_button = fin2d.tk.Button(top, textvariable=save_button_text, command=save_current_view)
        save_button.pack(side="right", padx=8, pady=4)
        fin2d.tk.Button(
            top, text="Export simulation surface...",
            command=lambda: self.export_surface_dialog(export_snapshot),
        ).pack(side="right", padx=8, pady=4)
        path_label.pack(side="left", fill="x", expand=True, padx=8, pady=4)

        fin2d.tk.Label(mode_bar, text="View:").pack(side="left", padx=(8, 3), pady=4)
        fin2d.tk.Radiobutton(
            mode_bar, text="Wireframe", variable=mode_var, value="wireframe",
            indicatoron=False, command=change_mode,
        ).pack(side="left", padx=2, pady=4)
        fin2d.tk.Radiobutton(
            mode_bar, text="High-resolution photo", variable=mode_var, value="photo",
            indicatoron=False, command=change_mode,
        ).pack(side="left", padx=2, pady=4)
        fin2d.tk.Label(mode_bar, text="Photographed side:").pack(side="left", padx=(16, 3), pady=4)
        fin2d.tk.Radiobutton(
            mode_bar, text="Left (-Z)", variable=side_var, value="left",
            indicatoron=False, command=select_side,
        ).pack(side="left", padx=2, pady=4)
        fin2d.tk.Radiobutton(
            mode_bar, text="Right (+Z)", variable=side_var, value="right",
            indicatoron=False, command=select_side,
        ).pack(side="left", padx=2, pady=4)
        flip_text = "yes" if texture_snapshot is not None and texture_snapshot["flip_applied"] else "no"
        fin2d.tk.Label(mode_bar, text=f"Image flipped for measurement: {flip_text}").pack(side="left", padx=16, pady=4)

        self._add_3d_view_slider(controls, "Pitch", pitch_var, -90.0, 90.0, schedule_render)
        self._add_3d_view_slider(controls, "Yaw", yaw_var, -180.0, 180.0, schedule_render)
        self._add_3d_view_slider(controls, "Roll", roll_var, -180.0, 180.0, schedule_render)
        display.bind("<Configure>", on_resize)
        mpl_widget.pack(side="top", fill="both", expand=True)
        render_wireframe()
        self.last_3d_view_figure = figure
        self.last_3d_view_controls = {
            "window": win,
            "mode": mode_var,
            "side": side_var,
            "pitch": pitch_var,
            "yaw": yaw_var,
            "roll": roll_var,
            "save_button": save_button,
            "save_button_text": save_button_text,
        }


def main() -> None:
    app = IntegratedFinShapeApp()
    if not fin2d.PIL_OK:
        fin2d.messagebox.showwarning(
            "Pillow required",
            "Pillow is required for photograph measurement and high-resolution 3D texture rendering.",
        )
    app.mainloop()


if __name__ == "__main__":
    main()
