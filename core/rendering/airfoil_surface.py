"""Continuous airfoil-style thickness reconstruction for dorsal-fin surfaces.

The measured planform remains authoritative. This module supplies only the
unmeasured thickness direction, using a closed symmetric NACA-style section at
each horizontal chord. The section shrinks with the measured chord and closes
at the leading edge, trailing edge, and fin apex.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from shapely import constrained_delaunay_triangles
from shapely.geometry import LineString, Point, Polygon

MAX_FULL_THICKNESS_RATIO = 0.12
NACA_CLOSED_COEFFICIENTS = (0.2969, -0.1260, -0.3516, 0.2843, -0.1036)
PHOTO_SPAN_STATIONS = 321
PHOTO_CHORD_STATIONS = 41
BOUNDARY_SPAN_STATIONS = 241
BOUNDARY_CHORD_STATIONS = 25
BOUNDARY_MIN_TRIANGLES = 5_000


def _naca_shape(chord_fraction) -> np.ndarray:
    """Return the unscaled closed NACA four-digit thickness ordinate."""

    u = np.clip(np.asarray(chord_fraction, dtype=float), 0.0, 1.0)
    a0, a1, a2, a3, a4 = NACA_CLOSED_COEFFICIENTS
    shape = a0 * np.sqrt(u) + a1 * u + a2 * u**2 + a3 * u**3 + a4 * u**4
    shape = np.where((u <= 0.0) | (u >= 1.0), 0.0, shape)
    return np.maximum(shape, 0.0)


_NACA_SHAPE_MAX = float(_naca_shape(np.linspace(0.0, 1.0, 100_001)).max())


def airfoil_max_full_thickness(
    chord: float,
    empirical_full_thickness: float,
    *,
    display_scale: float = 0.5,
    max_ratio: float = MAX_FULL_THICKNESS_RATIO,
) -> float:
    """Return a finite local maximum thickness that vanishes with chord."""

    chord = max(0.0, float(chord))
    empirical = max(0.0, float(empirical_full_thickness))
    if not np.isfinite([chord, empirical, display_scale, max_ratio]).all():
        raise ValueError("Airfoil thickness inputs must be finite.")
    if display_scale <= 0.0 or max_ratio <= 0.0:
        raise ValueError("Airfoil thickness scales must be positive.")
    return min(display_scale * empirical, max_ratio * chord)


def airfoil_full_thickness(
    chord_position,
    chord: float,
    maximum_full_thickness: float,
) -> np.ndarray:
    """Evaluate a closed symmetric airfoil's full thickness."""

    positions = np.asarray(chord_position, dtype=float)
    chord = float(chord)
    maximum = max(0.0, float(maximum_full_thickness))
    if not np.isfinite(positions).all() or not np.isfinite([chord, maximum]).all():
        raise ValueError("Airfoil section coordinates must be finite.")
    if chord <= 1e-12 or maximum <= 0.0:
        return np.zeros_like(positions)
    shape = _naca_shape(positions / chord)
    return maximum * shape / _NACA_SHAPE_MAX


def invert_airfoil_thickness(
    target_full_thickness: float,
    chord: float,
    maximum_full_thickness: float,
) -> tuple[float, float] | None:
    """Return the two chord positions having the requested full thickness."""

    target = float(target_full_thickness)
    chord = float(chord)
    maximum = float(maximum_full_thickness)
    if target < 0.0 or chord <= 1e-12 or maximum <= 0.0 or target > maximum + 1e-12:
        return None
    if target <= 1e-12:
        return 0.0, chord
    u = np.linspace(0.0, 1.0, 4097)
    if target >= maximum - 1e-12:
        peak = float(u[np.argmax(_naca_shape(u))])
        return peak * chord, peak * chord

    thickness = airfoil_full_thickness(u * chord, chord, maximum)
    peak = int(np.argmax(thickness))
    left = float(np.interp(target, thickness[:peak + 1], u[:peak + 1]))
    right = float(np.interp(target, thickness[peak:][::-1], u[peak:][::-1]))
    return left * chord, right * chord


def _line_intervals(geometry) -> list[tuple[float, float]]:
    if geometry.geom_type == "LineString":
        coordinates = np.asarray(geometry.coords, dtype=float)
        if len(coordinates) >= 2:
            return [(float(coordinates[:, 0].min()), float(coordinates[:, 0].max()))]
    if hasattr(geometry, "geoms"):
        return [interval for part in geometry.geoms for interval in _line_intervals(part)]
    return []


def _horizontal_intervals(polygon: Polygon, y: float) -> list[tuple[float, float]]:
    lo_x, _lo_y, hi_x, _hi_y = polygon.bounds
    span = max(hi_x - lo_x, 1.0)
    section = polygon.intersection(LineString([(lo_x - span, y), (hi_x + span, y)]))
    return sorted(
        (pair for pair in _line_intervals(section) if pair[1] - pair[0] > 1e-10),
        key=lambda pair: pair[0],
    )


def _containing_chord(
    polygon: Polygon,
    x: float,
    y: float,
    *,
    tolerance: float,
) -> tuple[float, float]:
    """Return the tissue interval containing one triangulated planform point."""

    intervals = _horizontal_intervals(polygon, y)
    containing = [
        pair for pair in intervals if pair[0] - tolerance <= x <= pair[1] + tolerance
    ]
    if len(containing) == 1:
        return containing[0]
    if containing:
        return min(containing, key=lambda pair: pair[1] - pair[0])

    # A triangulator can place a vertex exactly at a polygon ordinate where a
    # horizontal intersection degenerates to a Point. Boundary vertices have
    # already been handled; for an interior vertex, sample immediately above
    # and below and retain only an interval that still contains the vertex.
    for offset in (tolerance, -tolerance, 4.0 * tolerance, -4.0 * tolerance):
        nearby = [
            pair
            for pair in _horizontal_intervals(polygon, y + offset)
            if pair[0] - 8.0 * tolerance <= x <= pair[1] + 8.0 * tolerance
        ]
        if nearby:
            return min(nearby, key=lambda pair: pair[1] - pair[0])
    raise ValueError(
        f"Triangulated Photo-ID point ({x:.9g}, {y:.9g}) is outside its planform."
    )


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )


def _validated_planform(scene) -> tuple[np.ndarray, Polygon, float, float]:
    boundary = np.asarray(scene.outline, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1:] != (2,) or len(boundary) < 3:
        raise ValueError("Planform outline must contain at least three points.")
    polygon = Polygon(boundary)
    if not polygon.is_valid or polygon.area <= 0.0:
        raise ValueError("Planform outline must form a valid polygon.")

    base_y = 0.5 * float(boundary[0, 1] + boundary[-1, 1])
    scale = max(float(np.ptp(boundary[:, 0])), float(np.ptp(boundary[:, 1])), 1.0)
    tolerance = 1e-7 * scale
    if abs(float(boundary[0, 1] - boundary[-1, 1])) > tolerance:
        raise ValueError("Planform attachment endpoints must share one base height.")
    if polygon.bounds[1] < base_y - tolerance:
        raise ValueError("Planform contour extends materially below its attachment base.")
    return boundary, polygon, base_y, tolerance


def _single_chord_topology(
    boundary: np.ndarray,
    polygon: Polygon,
    base_y: float,
    tolerance: float,
    *,
    station_count: int,
) -> bool:
    """Return whether every interior span station is one connected chord."""

    top_y = float(boundary[:, 1].max())
    span = top_y - base_y
    if span <= tolerance:
        return False
    regular = np.linspace(
        base_y,
        top_y,
        max(129, min(1025, 2 * station_count)),
        endpoint=False,
    )[1:]
    y_values = boundary[:, 1]
    before = y_values[1:-1] - y_values[:-2]
    after = y_values[2:] - y_values[1:-1]
    turning = (before * after <= 0.0) & (
        (np.abs(before) > tolerance) | (np.abs(after) > tolerance)
    )
    critical_y = y_values[1:-1][turning]
    offsets = tolerance * np.array([-4.0, -1.0, 1.0, 4.0])
    topology_changes = np.column_stack((
        critical_y,
        critical_y[:, None] + offsets[None, :],
    )) if len(critical_y) else np.empty((0, 5), dtype=float)
    levels = np.unique(
        np.clip(
            np.concatenate((regular, topology_changes.ravel())),
            base_y + tolerance,
            top_y - tolerance,
        )
    )
    return all(len(_horizontal_intervals(polygon, float(y))) == 1 for y in levels)


def airfoil_mesh_strategy(scene, *, station_count: int = PHOTO_SPAN_STATIONS) -> str:
    """Choose the topology required by the measured Photo-ID planform."""

    boundary, polygon, base_y, tolerance = _validated_planform(scene)
    if _single_chord_topology(
        boundary,
        polygon,
        base_y,
        tolerance,
        station_count=station_count,
    ):
        return "structured-single-chord"
    return "boundary-multi-interval"


def _surface_stretch(vertices, faces, labels, textured_skin):
    selected = faces[labels == textured_skin]
    planar = vertices.copy()
    planar[:, 2] = 0.0
    planar_area = _triangle_areas(planar, selected)
    surface_area = _triangle_areas(vertices, selected)
    valid = planar_area > 1e-14
    if not np.any(valid):
        raise ValueError("Photo-ID skin triangulation has no finite planar area.")
    stretch = surface_area[valid] / planar_area[valid]
    weighted_mean = float(surface_area[valid].sum() / planar_area[valid].sum())
    return float(stretch.min()), weighted_mean, float(stretch.max())


def _readonly(values, dtype=None):
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _deduplicate_points(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Return stable unique points without collapsing measured edge detail."""

    scale = max(float(tolerance), 1e-12)
    seen = set()
    unique = []
    for point in np.asarray(points, dtype=float):
        key = tuple(np.rint(point / scale).astype(np.int64))
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
    return np.asarray(unique, dtype=float)


def _densify_closed_boundary(boundary: np.ndarray, maximum_spacing: float) -> np.ndarray:
    """Split long contour segments without moving any measured vertex."""

    dense = []
    for start, end in zip(boundary, np.roll(boundary, -1, axis=0)):
        distance = float(np.linalg.norm(end - start))
        steps = max(1, int(np.ceil(distance / maximum_spacing)))
        dense.extend(start + (end - start) * (offset / steps) for offset in range(steps))
    return np.asarray(dense, dtype=float)


def _subdivide_triangles(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split every triangle into four while sharing all edge midpoints."""

    vertices = [point.copy() for point in np.asarray(points, dtype=float)]
    midpoint_indices = {}

    def midpoint(a: int, b: int) -> int:
        edge = tuple(sorted((int(a), int(b))))
        if edge not in midpoint_indices:
            midpoint_indices[edge] = len(vertices)
            vertices.append(0.5 * (vertices[edge[0]] + vertices[edge[1]]))
        return midpoint_indices[edge]

    refined = []
    for a, b, c in np.asarray(faces, dtype=np.int64):
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return np.asarray(vertices, dtype=float), np.asarray(refined, dtype=np.int64)


def _subdivide_triangle_centroids(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each triangle into three without changing any existing edge."""

    vertices = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    centroids = vertices[faces].mean(axis=1)
    first_centroid = len(vertices)
    refined = []
    for index, (a, b, c) in enumerate(faces):
        center = first_centroid + index
        refined.extend(((a, b, center), (b, c, center), (c, a, center)))
    return (
        np.vstack((vertices, centroids)),
        np.asarray(refined, dtype=np.int64),
    )


def _dense_boundary_planform(
    boundary: np.ndarray,
    polygon: Polygon,
    base_y: float,
    tolerance: float,
    *,
    span_stations: int,
    chord_stations: int,
    minimum_triangles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a dense constrained mesh while preserving every contour edge."""

    clean_boundary = _deduplicate_points(boundary, tolerance * 0.1)
    if len(clean_boundary) < 3:
        raise ValueError("The Photo-ID boundary has fewer than three unique points.")
    if np.linalg.norm(clean_boundary[0] - clean_boundary[-1]) <= tolerance:
        clean_boundary = clean_boundary[:-1]

    top_y = float(clean_boundary[:, 1].max())
    width = float(clean_boundary[:, 0].max() - clean_boundary[:, 0].min())
    span_spacing = (top_y - base_y) / max(span_stations - 1, 1)
    chord_spacing = width / max(chord_stations - 1, 1)
    maximum_spacing = max(tolerance * 10.0, span_spacing, chord_spacing)
    dense_boundary = _densify_closed_boundary(clean_boundary, maximum_spacing)
    dense_polygon = Polygon(dense_boundary)
    if not dense_polygon.is_valid or dense_polygon.area <= tolerance * tolerance:
        raise ValueError("The dense Photo-ID contour is not a valid polygon.")

    triangles = constrained_delaunay_triangles(dense_polygon)
    if triangles.is_empty:
        raise ValueError("The dense Photo-ID planform contains no valid triangles.")

    scale = max(tolerance * 0.1, 1e-12)
    xy = [point.copy() for point in dense_boundary]
    index_by_key = {
        tuple(np.rint(point / scale).astype(np.int64)): index
        for index, point in enumerate(xy)
    }

    def point_index(point) -> int:
        values = np.asarray(point, dtype=float)
        key = tuple(np.rint(values / scale).astype(np.int64))
        index = index_by_key.get(key)
        if index is None:
            index = len(xy)
            xy.append(values)
            index_by_key[key] = index
        return index

    planar_faces = []
    for triangle in triangles.geoms:
        coordinates = np.asarray(triangle.exterior.coords[:3], dtype=float)
        face = np.asarray([point_index(point) for point in coordinates], dtype=np.int64)
        first = coordinates[1] - coordinates[0]
        second = coordinates[2] - coordinates[0]
        signed_area = first[0] * second[1] - first[1] * second[0]
        if abs(float(signed_area)) <= tolerance * tolerance:
            continue
        planar_faces.append(face if signed_area > 0.0 else face[[0, 2, 1]])
    if not planar_faces:
        raise ValueError("The dense Photo-ID planform contains no valid triangles.")

    xy = np.asarray(xy, dtype=float)
    planar_faces = np.asarray(planar_faces, dtype=np.int64)
    xy, planar_faces = _subdivide_triangles(xy, planar_faces)
    while len(planar_faces) * 4 <= minimum_triangles:
        xy, planar_faces = _subdivide_triangles(xy, planar_faces)
    if len(planar_faces) < minimum_triangles:
        xy, planar_faces = _subdivide_triangle_centroids(xy, planar_faces)

    boundary_count = len(dense_boundary)
    boundary_edges = np.column_stack(
        (
            np.arange(boundary_count, dtype=np.int64),
            np.roll(np.arange(boundary_count, dtype=np.int64), -1),
        )
    )
    return xy, planar_faces, boundary_edges


def _rebuild_structured_airfoil(
    scene,
    boundary: np.ndarray,
    polygon: Polygon,
    base_y: float,
    tolerance: float,
    *,
    display_scale: float,
    max_ratio: float,
    span_stations: int,
    chord_stations: int,
):
    """Build the continuous 1.3.8 span/chord surface for one-chord fins."""

    top_y = float(boundary[:, 1].max())
    levels = np.linspace(base_y, top_y, span_stations)[:-1]
    fractions = np.linspace(0.0, 1.0, chord_stations)
    rows = []
    for index, y in enumerate(levels):
        if index == 0:
            left, right = sorted((float(boundary[0, 0]), float(boundary[-1, 0])))
        else:
            intervals = _horizontal_intervals(polygon, float(y))
            if len(intervals) != 1:
                raise ValueError(
                    f"Structured Photo-ID section at Y={y:.9g} has "
                    f"{len(intervals)} tissue intervals."
                )
            left, right = intervals[0]
        chord = right - left
        empirical = 0.1191 * chord + 0.5164
        maximum = airfoil_max_full_thickness(
            chord, empirical, display_scale=display_scale, max_ratio=max_ratio
        )
        positions = fractions * chord
        x = right - positions
        half = 0.5 * airfoil_full_thickness(positions, chord, maximum)
        rows.append(np.column_stack((x, np.full(chord_stations, y), half)))

    positive = np.vstack(rows)
    apex_candidates = boundary[
        np.isclose(boundary[:, 1], top_y, rtol=0.0, atol=tolerance)
    ]
    apex_x = float(
        apex_candidates[:, 0].mean()
        if len(apex_candidates)
        else boundary[int(np.argmax(boundary[:, 1])), 0]
    )
    positive = np.vstack((positive, np.array([[apex_x, top_y, 0.0]])))
    skin_vertices = len(positive)
    negative = positive.copy()
    negative[:, 2] *= -1.0
    vertices = np.vstack((positive, negative))

    positive_faces = []
    row_count = len(levels)
    for row in range(row_count - 1):
        start = row * chord_stations
        next_start = (row + 1) * chord_stations
        for column in range(chord_stations - 1):
            a = start + column
            b = a + 1
            c = next_start + column
            d = c + 1
            positive_faces.extend(((a, c, b), (b, c, d)))
    apex = skin_vertices - 1
    last = (row_count - 1) * chord_stations
    for column in range(chord_stations - 1):
        positive_faces.append((last + column, apex, last + column + 1))

    positive_faces = np.asarray(positive_faces, dtype=np.int64)
    negative_faces = positive_faces[:, ::-1] + skin_vertices
    base_faces = []
    for column in range(chord_stations - 1):
        a, b = column, column + 1
        base_faces.extend(
            ((a, b, a + skin_vertices), (b, b + skin_vertices, a + skin_vertices))
        )
    base_faces = np.asarray(base_faces, dtype=np.int64)
    faces = np.vstack((positive_faces, negative_faces, base_faces))
    labels = np.asarray(
        ["positive_z_skin"] * len(positive_faces)
        + ["negative_z_skin"] * len(negative_faces)
        + ["base_attachment"] * len(base_faces)
    )
    colors = np.empty((len(faces), 4), dtype=float)
    colors[:] = (0.82, 0.83, 0.84, 1.0)
    colors[labels == "base_attachment"] = (0.46, 0.48, 0.50, 1.0)

    minimum, mean, maximum = _surface_stretch(
        vertices, faces, labels, scene.textured_skin
    )
    return replace(
        scene,
        vertices=_readonly(vertices, float),
        faces=_readonly(faces, np.int64),
        face_labels=_readonly(labels, str),
        face_colors=_readonly(colors, float),
        surface_stretch_min=minimum,
        surface_stretch_mean=mean,
        surface_stretch_max=maximum,
    )


def _rebuild_boundary_airfoil(
    scene,
    boundary: np.ndarray,
    polygon: Polygon,
    base_y: float,
    tolerance: float,
    *,
    display_scale: float,
    max_ratio: float,
    span_stations: int,
    chord_stations: int,
    boundary_min_triangles: int,
):
    """Build and lift a dense boundary-conforming multi-interval surface."""

    boundary_span = max(65, min(int(span_stations), BOUNDARY_SPAN_STATIONS))
    boundary_chord = max(13, min(int(chord_stations), BOUNDARY_CHORD_STATIONS))
    xy, planar_faces, boundary_edges = _dense_boundary_planform(
        boundary,
        polygon,
        base_y,
        tolerance,
        span_stations=boundary_span,
        chord_stations=boundary_chord,
        minimum_triangles=boundary_min_triangles,
    )
    skin_vertices = len(xy)
    positive = np.column_stack((xy, np.zeros(skin_vertices, dtype=float)))

    base_left, base_right = sorted((float(boundary[0, 0]), float(boundary[-1, 0])))
    polygon_boundary = polygon.boundary
    half_thickness = np.zeros(skin_vertices, dtype=float)
    for index, (x, y, _z) in enumerate(positive):
        point = Point(float(x), float(y))
        on_base = abs(float(y) - base_y) <= tolerance
        on_outline = polygon_boundary.distance(point) <= tolerance
        if on_outline and not on_base:
            continue
        if on_base:
            left, right = base_left, base_right
        else:
            left, right = _containing_chord(
                polygon, float(x), float(y), tolerance=tolerance
            )
        chord = right - left
        empirical = 0.1191 * chord + 0.5164
        maximum = airfoil_max_full_thickness(
            chord, empirical, display_scale=display_scale, max_ratio=max_ratio
        )
        chord_position = float(np.clip(right - x, 0.0, chord))
        half_thickness[index] = 0.5 * float(
            airfoil_full_thickness(chord_position, chord, maximum)
        )

    positive[:, 2] = half_thickness
    negative = positive.copy()
    negative[:, 2] *= -1.0
    vertices = np.vstack((positive, negative))

    positive_faces = planar_faces
    negative_faces = planar_faces[:, ::-1] + skin_vertices
    wall_faces = []
    wall_labels = []
    for a, b in boundary_edges:
        wall_faces.extend(
            ((a, a + skin_vertices, b), (b, a + skin_vertices, b + skin_vertices))
        )
        label = (
            "base_attachment"
            if abs(float(xy[a, 1]) - base_y) <= tolerance
            and abs(float(xy[b, 1]) - base_y) <= tolerance
            else "edge_wall"
        )
        wall_labels.extend((label, label))
    wall_faces = np.asarray(wall_faces, dtype=np.int64)
    faces = np.vstack((positive_faces, negative_faces, wall_faces))
    labels = np.asarray(
        ["positive_z_skin"] * len(positive_faces)
        + ["negative_z_skin"] * len(negative_faces)
        + wall_labels
    )
    colors = np.empty((len(faces), 4), dtype=float)
    colors[:] = (0.82, 0.83, 0.84, 1.0)
    colors[labels == "edge_wall"] = (0.30, 0.32, 0.34, 1.0)
    colors[labels == "base_attachment"] = (0.46, 0.48, 0.50, 1.0)
    minimum, mean, maximum = _surface_stretch(
        vertices, faces, labels, scene.textured_skin
    )
    return replace(
        scene,
        vertices=_readonly(vertices, float),
        faces=_readonly(faces, np.int64),
        face_labels=_readonly(labels, str),
        face_colors=_readonly(colors, float),
        surface_stretch_min=minimum,
        surface_stretch_mean=mean,
        surface_stretch_max=maximum,
    )


def rebuild_texture_scene_airfoil(
    scene,
    *,
    display_scale: float = 0.5,
    max_ratio: float = MAX_FULL_THICKNESS_RATIO,
    span_stations: int = PHOTO_SPAN_STATIONS,
    chord_stations: int = PHOTO_CHORD_STATIONS,
    boundary_min_triangles: int = BOUNDARY_MIN_TRIANGLES,
):
    """Build the smoothest valid airfoil topology for the measured planform.

    Ordinary fins use the dense structured span/chord surface introduced in
    1.3.8. It prevents coarse arbitrary triangles from becoming the silhouette
    at oblique views. A boundary-conforming fallback is reserved for contours
    whose horizontal sections genuinely contain multiple tissue intervals.
    """

    if type(span_stations) is not int or span_stations < 33:
        raise ValueError("Airfoil reconstruction needs at least 33 span stations.")
    if type(chord_stations) is not int or chord_stations < 9:
        raise ValueError("Airfoil reconstruction needs at least 9 chord stations.")
    if type(boundary_min_triangles) is not int or boundary_min_triangles < 32:
        raise ValueError("Boundary reconstruction needs at least 32 triangles.")
    boundary, polygon, base_y, tolerance = _validated_planform(scene)
    if _single_chord_topology(
        boundary,
        polygon,
        base_y,
        tolerance,
        station_count=span_stations,
    ):
        return _rebuild_structured_airfoil(
            scene,
            boundary,
            polygon,
            base_y,
            tolerance,
            display_scale=display_scale,
            max_ratio=max_ratio,
            span_stations=span_stations,
            chord_stations=chord_stations,
        )
    return _rebuild_boundary_airfoil(
        scene,
        boundary,
        polygon,
        base_y,
        tolerance,
        display_scale=display_scale,
        max_ratio=max_ratio,
        span_stations=span_stations,
        chord_stations=chord_stations,
        boundary_min_triangles=boundary_min_triangles,
    )
