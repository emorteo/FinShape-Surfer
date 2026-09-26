from pathlib import Path
from typing import List

import pandas as pd

from core.io.data_loader import read_table
from core.reconstruction.wireframe import build_landmarks, build_controls, compute_2d_curves
from core.ui.plotting_3d import plot_2d, plot_3d_wireframe, make_montage, VERSION


def run(input_path: Path, outdir: Path, bc30scale: float, y_slices: int, z_slices: int,
        top_focus_power: float, montage: bool) -> None:
    """Run the reconstruction pipeline on a table of ratio measurements.

    This is a lightweight compatibility wrapper used by tests and external
    scripts. It mirrors the historical `FinShape3D` CLI behaviour.
    """
    df = read_table(input_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outs2d: List[Path] = []
    outs3d: List[Path] = []

    for i, row in df.iterrows():
        fin_id = str(row.get("id", f"row{i}"))
        lm, u_AB = build_landmarks(row, BC30_scale=bc30scale)
        ctrl = build_controls(lm, u_AB)
        curves2d = compute_2d_curves(ctrl, n=900)

        p2d = outdir / f"FinShape3D_{VERSION}_2D_{fin_id}.png"
        p3d = outdir / f"FinShape3D_{VERSION}_3D_{fin_id}.png"

        # save annotated 2D and 3D wireframe images
        plot_2d(curves2d, lm, ctrl, fin_id, p2d, legend_anchor_xy=(-4.0, 4.0))
        plot_3d_wireframe(curves2d, fin_id, p3d, n_y_slices=y_slices, n_chord_pts=90,
                          n_z_slices=z_slices, top_focus_power=top_focus_power)

        outs2d.append(p2d)
        outs3d.append(p3d)

    if montage and outs2d:
        make_montage(outs2d, outdir / f"FinShape3D_{VERSION}_montage_2D.png",
                     title=f"FinShape3D {VERSION} — 2D Contours")
    if montage and outs3d:
        make_montage(outs3d, outdir / f"FinShape3D_{VERSION}_montage_3D.png",
                     title=f"FinShape3D {VERSION} — 3D Wireframes")
