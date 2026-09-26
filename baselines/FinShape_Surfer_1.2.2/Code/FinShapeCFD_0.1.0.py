#!/usr/bin/env python3
"""Experimental surface preparation, not a CFD or bioheat solver.

Input is the fitted AB=10 contour, not pixels of the wireframe PNG.
Requires NumPy, Shapely >=2.1 (GEOS >=3.10), and trimesh.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
import zipfile

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
import trimesh

VERSION = "0.1.0"
MODEL_PATH = Path(__file__).with_name("FinShape3D_1.1.20.py")
ASSUMPTION = (
    "Minimum tip/trailing-edge thickness was imposed to prevent geometric collapse near the apex. "
    "The same floor applies to the leading edge. This is a structural visualization assumption "
    "explicitly promoted to an experimental surface scenario, not measured biological thickness. "
    "The original units and empirical range of the inherited SP/T equations are unresolved. "
    "Physical scaling does not resolve that uncertainty."
)


@dataclass(frozen=True)
class SurfaceSettings:
    ab_length_m: float
    scale_provenance: str
    accept_experimental_thickness: bool = False
    edge_min_thickness_units: float = 0.05
    subdivisions: int = 2

    def validate(self):
        if not np.isfinite(self.ab_length_m) or self.ab_length_m <= 0:
            raise ValueError("Physical AB length must be finite and greater than zero, in metres.")
        if not isinstance(self.scale_provenance, str) or not self.scale_provenance.strip():
            raise ValueError("State how physical AB was measured or that it is an assumed scenario.")
        if self.accept_experimental_thickness is not True:
            raise ValueError("Explicit acceptance of the experimental thickness assumption is required.")
        if not np.isfinite(self.edge_min_thickness_units) or self.edge_min_thickness_units <= 0:
            raise ValueError("Minimum thickness must be finite and positive in AB=10 coordinate units.")
        if type(self.subdivisions) is not int or not 1 <= self.subdivisions <= 3:
            raise ValueError("Uniform triangle subdivisions must be an integer from 1 to 3.")


def load_model():
    name = "finshape_cfd_model_1_1_20"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, MODEL_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def make_input(curves, landmarks, settings, provenance=None):
    """Freeze normalized coordinates and provenance; no image data is embedded."""
    settings.validate()
    return {
        "schema": "finshape.surface-input/1",
        "coordinates": "X chord, Y height, Z thickness; right-handed; B origin; AB=10",
        "curves": {k: np.asarray(v, dtype=float).tolist() for k, v in curves.items()},
        "landmarks": {k: np.asarray(v, dtype=float).tolist() for k, v in landmarks.items()},
        "settings": asdict(settings),
        "provenance": provenance or {},
    }


def _polygon_from_input(data):
    if data.get("schema") != "finshape.surface-input/1":
        raise ValueError("Unsupported surface input schema.")
    pts = np.asarray(data["curves"]["outline"], dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or not 3 <= len(pts) <= 5000:
        raise ValueError("Expected 3 to 5000 two-dimensional outline samples.")
    if not np.isfinite(pts).all():
        raise ValueError("Outline contains NaN or infinity.")
    lm = {k: np.asarray(v, dtype=float) for k, v in data["landmarks"].items()}
    for name, p in lm.items():
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError(f"Invalid landmark {name}.")
    if not all(k in lm for k in ("A", "B", "C30")):
        raise ValueError("A, B and C30 are required to verify AB=10 normalization.")
    if not np.allclose(lm["B"], [0, 0], rtol=0, atol=1e-8):
        raise ValueError("B must be the origin.")
    if not np.isclose(np.linalg.norm(lm["A"] - lm["B"]), 10, rtol=0, atol=1e-7):
        raise ValueError("Input must have AB=10; arbitrary ratio-scaffold scales are not accepted.")
    if abs(lm["C30"][1]) > 1e-8 or lm["C30"][0] >= 0:
        raise ValueError("C30 must be on the negative X axis at the base.")
    if not np.allclose(pts[0], lm["B"], rtol=0, atol=1e-8) or not np.allclose(pts[-1], lm["C30"], rtol=0, atol=1e-8):
        raise ValueError("Outline must follow B to A to C30; closure is the C30-B base.")
    if pts[:, 1].min() < -1e-8 or pts[:, 1].max() <= 0:
        raise ValueError("Contour crosses below the base or has no positive height. Review the fit.")
    # Only remove consecutive identical samples; never simplify or repair the fit.
    pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]
    polygon = Polygon(pts)
    if not polygon.is_valid or polygon.area <= 1e-10:
        raise ValueError(f"Invalid fitted outline: {shapely.is_valid_reason(polygon)}. No automatic repair applied.")
    for name in ("A", "B", "C30"):
        if np.min(np.linalg.norm(pts - lm[name], axis=1)) > 1e-7:
            raise ValueError(f"Required landmark {name} is absent from the sampled boundary.")
    return polygon, pts, lm


def _triangulate(polygon, subdivisions):
    """GEOS constrained triangulation, followed by conforming shared-edge subdivision."""
    # Partition the interior into square cells before constrained triangulation.
    # This prevents long fans from thousands of closely spaced boundary samples.
    lo_x, lo_y, hi_x, hi_y = polygon.bounds
    spacing = max(hi_x - lo_x, hi_y - lo_y) / 32.0
    xs = np.linspace(lo_x, hi_x, max(2, int(np.ceil((hi_x - lo_x) / spacing)) + 1))
    ys = np.linspace(lo_y, hi_y, max(2, int(np.ceil((hi_y - lo_y) / spacing)) + 1))
    cells = [shapely.box(x0, y0, x1, y1) for x0, x1 in zip(xs[:-1], xs[1:]) for y0, y1 in zip(ys[:-1], ys[1:])]
    clipped = shapely.intersection(polygon, cells)
    triangulated = shapely.constrained_delaunay_triangles(clipped)
    triangles = [tri for collection in triangulated for tri in collection.geoms]
    if not triangles:
        raise ValueError("Constrained triangulation produced no faces.")
    covered = shapely.union_all(triangles)
    tolerance = max(1e-10, polygon.area * 1e-10)
    if polygon.symmetric_difference(covered).area > tolerance or abs(sum(t.area for t in triangles) - covered.area) > tolerance:
        raise ValueError("Triangulation coverage or overlap validation failed.")
    # Keep original coordinates at boundary samples; coalesce only floating-point
    # duplicates of intersections computed from neighboring cells (1e-12 units).
    vertices = [[x, y, 0.0] for x, y in list(polygon.exterior.coords)[:-1]]
    faces = []
    lookup = {tuple(np.round(v[:2], 12)): i for i, v in enumerate(vertices)}
    for tri in triangles:
        face = []
        for xy in list(tri.exterior.coords)[:3]:
            key = tuple(np.round(xy, 12))
            if key not in lookup:
                lookup[key] = len(vertices)
                vertices.append([*xy, 0.0])
            face.append(lookup[key])
        if len(set(face)) == 3:
            faces.append(face)
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    signed = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])[:, 2]
    faces[signed < 0] = faces[signed < 0, ::-1]
    faces = faces[signed != 0]
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(unique)]
    for _ in range(subdivisions):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[inverse] == 1]
    if not len(boundary) or np.any(counts > 2):
        raise ValueError("Invalid planar triangulation topology.")
    return vertices[:, :2], faces, boundary


def _intervals(geometry):
    if geometry.geom_type == "LineString":
        x = np.asarray(geometry.coords)[:, 0]
        return [(float(x.min()), float(x.max()))]
    if hasattr(geometry, "geoms"):
        return [pair for part in geometry.geoms for pair in _intervals(part)]
    return []


def _thickness(xy, boundary, polygon, floor):
    """Lift the complete planform, including overhangs above A, without a chord cutoff."""
    model = load_model()
    lo, _, hi, _ = polygon.bounds
    thickness = np.full(len(xy), floor, dtype=float)
    side_vertices = np.zeros(len(xy), dtype=bool)
    for a, b in boundary:
        # The C30-B closure is an attachment face, not a zero-thickness skin edge.
        if not (abs(xy[a, 1]) < 1e-9 and abs(xy[b, 1]) < 1e-9):
            side_vertices[[a, b]] = True
    cache = {}
    for i, (x, y) in enumerate(xy):
        if side_vertices[i]:
            continue
        key = 0.0 if abs(y) <= 1e-9 else float(y)
        if key not in cache:
            if key == 0.0:
                # Normalization can leave C30 a few ulps above/below Y=0.
                # Evaluate attachment thickness on the defined B-C30 chord.
                base = polygon.exterior.coords
                cache[key] = [tuple(sorted((base[0][0], base[-2][0])))]
            else:
                cache[key] = _intervals(polygon.intersection(LineString([(lo - 1, key), (hi + 1, key)])))
        candidates = [(left, right) for left, right in cache[key] if left - 1e-9 <= x <= right + 1e-9 and right - left > 1e-12]
        if len(candidates) != 1:
            raise ValueError("Ambiguous horizontal chord at an interior vertex; review the fitted contour.")
        left, right = candidates[0]
        chord = right - left
        sp, tm = model.morteo_SP_T(chord)
        sp = model.clamp(sp, 0.05 * chord, 0.95 * chord)
        thickness[i] = model.structural_visual_thickness(
            np.array([np.clip(right - x, 0, chord)]), chord, sp, max(tm, 1e-4), edge_min_thickness=floor,
        )[0]
    if not np.isfinite(thickness).all() or np.any(thickness <= 0):
        raise ValueError("Thickness must remain finite and strictly positive.")
    return thickness


def validate_surface(mesh):
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Nonfinite surface coordinates.")
    area = mesh.area_faces
    if np.any(area <= 0) or not np.isfinite(area).all():
        raise ValueError("Degenerate surface triangles.")
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.euler_number != 2 or not np.isfinite(mesh.volume) or mesh.volume <= 0:
        raise ValueError("Surface failed closure, winding, Euler characteristic or positive-volume validation.")
    # Closure alone does not establish connectedness.
    adjacency = [[] for _ in mesh.vertices]
    for a, b in mesh.edges_unique:
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen, pending = {0}, [0]
    while pending:
        for vertex in adjacency[pending.pop()]:
            if vertex not in seen:
                seen.add(vertex)
                pending.append(vertex)
    if len(seen) != len(mesh.vertices):
        raise ValueError("Surface has disconnected or unreferenced vertices.")
    angles = np.degrees(mesh.face_angles)
    return {
        "watertight": True, "consistent_outward_winding": True, "connected_components": 1,
        "euler_characteristic": int(mesh.euler_number), "vertices": len(mesh.vertices), "triangles": len(mesh.faces),
        "surface_area_m2": float(mesh.area), "enclosed_volume_m3": float(mesh.volume),
        "bounds_m": mesh.bounds.tolist(), "minimum_triangle_angle_deg": float(angles.min()),
        "triangles_below_5_degrees": int(np.count_nonzero(angles.min(axis=1) < 5)),
        "quality_note": "5 degrees is a reporting flag, not a CFD acceptance threshold; solver remeshing and quality checks remain required.",
    }


def build_surface(data):
    settings = SurfaceSettings(**data["settings"])
    settings.validate()
    polygon, input_points, landmarks = _polygon_from_input(data)
    xy, planar_faces, boundary = _triangulate(polygon, settings.subdivisions)
    thickness = _thickness(xy, boundary, polygon, settings.edge_min_thickness_units)
    n = len(xy)
    vertices = np.vstack([np.column_stack([xy, thickness / 2]), np.column_stack([xy, -thickness / 2])])
    faces = [*planar_faces.tolist(), *(planar_faces[:, ::-1] + n).tolist()]
    labels = ["positive_z_skin"] * len(planar_faces) + ["negative_z_skin"] * len(planar_faces)
    for a, b in boundary:
        faces.extend([[int(a), int(a + n), int(b)], [int(b), int(a + n), int(b + n)]])
        label = "base_attachment" if abs(xy[a, 1]) < 1e-9 and abs(xy[b, 1]) < 1e-9 else "edge_wall"
        labels.extend([label, label])
    scale = settings.ab_length_m / 10.0
    mesh = trimesh.Trimesh(vertices=vertices * scale, faces=np.asarray(faces), process=False)
    report = validate_surface(mesh)
    # A conforming nonoverlapping planar triangulation lifted to +/- positive
    # continuous thickness cannot cross itself. This certifies this construction,
    # not arbitrary imported STL files.
    report.update({
        "self_intersection_check": "Construction certificate: valid simple polygon, checked nonoverlapping coverage, shared triangulation, strictly positive thickness, boundary-only walls.",
        "planform_area_m2": polygon.area * scale**2,
        "input_boundary_samples": len(input_points),
        "interior_triangulation": "32-cell grid partition followed by GEOS constrained Delaunay and uniform subdivisions; 1e-12 unit merge tolerance",
        "boundary_samples_preserved": all(np.any(np.all(xy == p, axis=1)) for p in input_points),
        "apex_height_m": float(mesh.vertices[:, 1].max()),
        "minimum_full_thickness_m": float(thickness.min() * scale),
        "landmark_boundary_residuals_units": {k: float(polygon.boundary.distance(shapely.Point(p))) for k, p in landmarks.items()},
        "simulation_ready": False,
    })
    if not report["boundary_samples_preserved"]:
        raise ValueError("Triangulation omitted input boundary samples.")
    return mesh, np.asarray(labels), report


def simulation_template(data, report):
    return {
        "schema": "finshape.simulation-preparation/1", "status": "INCOMPLETE_NOT_SOLVER_INPUT",
        "surface": "surface.stl", "units": "metres", "axes": "X chord; Y height; Z thickness (different from Matplotlib display order X,Z,Y)",
        "boundary_groups": "patches.json and surface.obj; groups are geometric, not anatomical left/right",
        "hydrodynamics": {
            "fluid_density_kg_m3": None, "dynamic_viscosity_Pa_s": None,
            "velocity_vector_m_s": None, "reference_length_m": None, "reference_area_m2": None,
            "reference_area_definition": None, "angle_of_attack_deg": None, "yaw_deg": None,
            "reynolds_number": None, "turbulence_model": None, "wall_treatment": None,
            "inlet_turbulence": None, "domain_extents_m": None, "boundary_conditions": None,
            "body_attachment_or_isolated_fin": None, "solver_and_version": None,
            "convergence_criteria": None, "volume_mesh_and_boundary_layers": None,
        },
        "thermal": {
            "analysis_type": None, "water_temperature_K": None, "base_boundary_condition": None,
            "initial_tissue_temperature_K": None, "tissue_conductivity_W_m_K": None,
            "tissue_density_kg_m3": None, "tissue_heat_capacity_J_kg_K": None,
            "blood_temperature_K": None, "blood_density_kg_m3": None, "blood_heat_capacity_J_kg_K": None,
            "perfusion_rate_s_inv": None, "vascular_or_countercurrent_model": None,
            "metabolic_heat_W_m3": None, "fluid_solid_interface_coupling": None,
            "note": "Choose and justify conduction/CHT/bioheat model. Perfusion, vasculature and heat transfer are not inferred from area.",
        },
        "required_studies": ["At least three volume-mesh resolutions and discretization uncertainty", "Domain and boundary-layer sensitivity", "Thickness and imposed-edge-floor sensitivity", "Physical-scale uncertainty", "Thermal/perfusion validation against observations"],
        "geometric_measurements_only": {k: report[k] for k in ("surface_area_m2", "enclosed_volume_m3", "planform_area_m2")},
        "assumption": ASSUMPTION,
    }



def surface_preview(mesh):
    """Render the exported geometry itself, not the preceding wireframe."""
    model = load_model()
    model.ensure_matplotlib()
    fig = model.plt.figure(figsize=(9, 7))
    try:
        ax = fig.add_subplot(111, projection="3d")
        x, height, thickness = mesh.vertices.T
        ax.plot_trisurf(x, thickness, height, triangles=mesh.faces, color="#91b7bf", linewidth=0, antialiased=True)
        ax.set_xlabel("X chord (m)")
        ax.set_ylabel("Z thickness (m)")
        ax.set_zlabel("Y height (m)")
        ax.set_yticks([float(thickness.min()), float(thickness.max())])
        ax.tick_params(labelsize=8, pad=1)
        extent = np.ptp(mesh.vertices, axis=0)
        ax.set_box_aspect(extent[[0, 2, 1]])
        ax.view_init(elev=20, azim=-65)
        ax.set_title("Experimental closed surface")
        fig.text(0.5, 0.03, "Imposed minimum edge/tip thickness; biological dimensions unresolved.", ha="center", fontsize=9)
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=140)
        return output.getvalue()
    finally:
        model.plt.close(fig)


def export_package(data, destination, *, overwrite=False):
    """Validate first, serialize and reload both surfaces, then publish one ZIP."""
    destination = Path(destination).expanduser().resolve()
    if destination.suffix.lower() != ".zip":
        raise ValueError("Surface package filename must end in .zip.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    print(f"[STARTUP] Surface package directory: {destination.parent}")
    mesh, labels, report = build_surface(data)
    members = {"input.json": _json_bytes(data)}
    members["surface.stl"] = mesh.export(file_type="stl")
    # OBJ supports named geometric boundary groups. Face indices also live in JSON.
    obj = ["# FinShape experimental surface; units: metres; X chord Y height Z thickness"]
    obj.extend("v " + " ".join(format(float(x), ".17g") for x in vertex) for vertex in mesh.vertices)
    previous = None
    for face, label in zip(mesh.faces, labels):
        if label != previous:
            obj.append(f"g {label}")
            previous = label
        obj.append("f " + " ".join(str(int(i) + 1) for i in face))
    members["surface.obj"] = ("\n".join(obj) + "\n").encode("ascii")
    for suffix in ("stl", "obj"):
        restored = trimesh.load(io.BytesIO(members[f"surface.{suffix}"]), file_type=suffix, force="mesh", process=True)
        validate_surface(restored)
        if len(restored.faces) != len(mesh.faces) or not np.isclose(restored.volume, mesh.volume, rtol=1e-5, atol=0):
            raise ValueError(f"{suffix.upper()} round-trip changed triangle count or volume.")
    report["stl_obj_roundtrip"] = "PASS: closure, winding, volume (rtol 1e-5), face count"
    members["validation.json"] = _json_bytes(report)
    members["surface_preview.png"] = surface_preview(mesh)
    groups = {}
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        groups[str(label)] = {"face_indices_zero_based": ids.tolist(), "area_m2": float(mesh.area_faces[ids].sum())}
    members["patches.json"] = _json_bytes({"index_order": "surface.obj and mesh.npz (not reordered STL imports)", "groups": groups,
        "base_note": "base_attachment closes the solid; it is not automatically a wetted wall or a thermal boundary condition."})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, vertices_m=mesh.vertices, faces=mesh.faces, face_group=labels)
    members["mesh.npz"] = buffer.getvalue()
    members["simulation_setup.json"] = _json_bytes(simulation_template(data, report))
    members["README.md"] = (
        "# FinShape experimental simulation surface\n\n"
        "Extract this archive. surface.stl is a closed surface in metres; surface.obj also names patches. "
        "mesh.npz retains indexed coordinates, faces and group names. input.json freezes the complete fitted outline, "
        "landmarks, scale and reconstruction settings for replay. validation.json reports geometric checks.\n\n"
        + ASSUMPTION + "\n\n"
        "The measured XY planform is preserved as sampled, including its apex. No artificial tip chord widening is used "
        "in this export. Nonzero edge walls and the base close the solid. The volume is a modeled tissue envelope, "
        "without internal vasculature. Normals point out of the solid (not out of a surrounding fluid domain).\n\n"
        "This is not a volume mesh or a runnable simulation. Complete simulation_setup.json from verified data. "
        "Choose base/body attachment, domain, fluid properties, boundary conditions, turbulence and thermal model. "
        "Remesh for the chosen solver, add boundary layers, check resolution at the finite edge, and run convergence studies. "
        "STL has no unit metadata; configure metres explicitly on import. OBJ groups and patches.json preserve attachment semantics.\n\n"
        "For an OpenFOAM installation that supports this utility: `surfaceCheck -checkSelfIntersection surface.stl`. "
        "Then construct a solver-specific domain and volume mesh and run `checkMesh`. These commands were not executed by this exporter.\n\n"
        "Replay: `python FinShapeCFD_0.1.0.py --replay original.zip --output reproduced.zip`. "
        "Compare geometric arrays, not archive timestamps. This export contains local provenance paths and catalog IDs; review before sharing.\n\n"
        "Sources: [project surface/CFD protocol, sections 7.7-7.8]; "
        "[OpenFOAM meshing](https://doc.cfd.direct/openfoam/user-guide-v13/snappyhexmesh); "
        "[Shapely constrained triangulation](https://shapely.readthedocs.io/en/stable/reference/shapely.constrained_delaunay_triangles.html).\n"
    ).encode("utf-8")
    members["manifest.json"] = _json_bytes({
        "exporter_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXPERIMENTAL_SURFACE_PREPARATION", "simulation_ready": False,
        "assumptions": [ASSUMPTION, "Symmetric thickness; complete boundary walls; planar base closure; no tissue or flow solved."],
        "units": "m", "model": MODEL_PATH.name,
        "source_sha256": {Path(__file__).name: _sha(Path(__file__).read_bytes()), MODEL_PATH.name: _sha(MODEL_PATH.read_bytes())},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "shapely": shapely.__version__, "GEOS": shapely.geos_version_string, "trimesh": trimesh.__version__},
        "settings": data["settings"], "files_sha256": {k: _sha(v) for k, v in members.items()},
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip", delete=False) as temp:
            temp_path = Path(temp.name)
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        if overwrite:
            temp_path.replace(destination)
        else:
            # Hard-link publication is atomic and refuses a concurrent overwrite.
            import os
            os.link(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    print(f"[DONE] Experimental surface package: {destination}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="surface-input/1 JSON; coordinates normalized to AB=10")
    source.add_argument("--replay", type=Path, help="Previously exported package; use its recorded input/settings")
    parser.add_argument("--output", required=True, type=Path, help="New .zip package, never silently overwritten")
    args = parser.parse_args(argv)
    try:
        if args.replay:
            with zipfile.ZipFile(args.replay) as archive:
                if archive.getinfo("input.json").file_size > 20_000_000:
                    raise ValueError("Input exceeds 20 MB limit.")
                raw = archive.read("input.json")
                if archive.getinfo("manifest.json").file_size > 1_000_000:
                    raise ValueError("Manifest exceeds size limit.")
                manifest = json.loads(archive.read("manifest.json"))
                if _sha(raw) != manifest["files_sha256"]["input.json"]:
                    raise ValueError("Recorded input checksum does not match.")
                for name, path in ((Path(__file__).name, Path(__file__)), (MODEL_PATH.name, MODEL_PATH)):
                    if manifest["source_sha256"].get(name) != _sha(path.read_bytes()):
                        raise ValueError("Replay source version/hash differs from the original exporter.")
                data = json.loads(raw)
        else:
            if args.input.stat().st_size > 20_000_000:
                raise ValueError("Input exceeds 20 MB limit.")
            data = json.loads(args.input.read_text(encoding="utf-8"))
        export_package(data, args.output)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Surface export failed: {exc}\n")


if __name__ == "__main__":
    main()

