from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from PIL import Image
from shapely import constrained_delaunay_triangles, get_parts
from shapely.geometry import Point, Polygon

from core.rendering.airfoil_surface import (
    airfoil_full_thickness,
    airfoil_max_full_thickness,
    airfoil_mesh_strategy,
    rebuild_texture_scene_airfoil,
)
from core.rendering.photo_display import render_smooth_photo_frame
from core.rendering.photo_mask import PhotoMaskTextureProxy, photo_mask_snapshot
from finshape_surfer_gui import _HQDisplayProxy


@dataclass(frozen=True)
class Scene:
    vertices: np.ndarray
    faces: np.ndarray
    face_labels: np.ndarray
    face_colors: np.ndarray
    outline: np.ndarray
    textured_skin: str = "positive_z_skin"
    surface_stretch_min: float = 1.0
    surface_stretch_mean: float = 1.0
    surface_stretch_max: float = 1.0


def _scene_from_outline(outline):
    outline = np.asarray(outline, dtype=float)
    polygon = Polygon(outline)
    points = []
    point_index = {}
    planar_faces = []
    for triangle in get_parts(constrained_delaunay_triangles(polygon)):
        face = []
        for coordinate in np.asarray(triangle.exterior.coords[:-1], dtype=float):
            key = tuple(np.round(coordinate, 12))
            if key not in point_index:
                point_index[key] = len(points)
                points.append(coordinate)
            face.append(point_index[key])
        planar_faces.append(face)
    xy = np.asarray(points, dtype=float)
    positive = np.column_stack((xy, np.full(len(xy), 0.1)))
    negative = np.column_stack((xy, np.full(len(xy), -0.1)))
    planar_faces = np.asarray(planar_faces, dtype=int)
    faces = np.vstack((planar_faces, planar_faces[:, ::-1] + len(xy)))
    labels = np.asarray(
        ["positive_z_skin"] * len(planar_faces)
        + ["negative_z_skin"] * len(planar_faces)
    )
    return Scene(
        np.vstack((positive, negative)),
        faces,
        labels,
        np.ones((len(faces), 4)),
        outline,
    )


def _coarse_scene():
    return _scene_from_outline([
        [0.0, 0.0], [-0.3, 1.5], [-0.8, 3.0], [-1.6, 4.0],
        [-2.4, 3.0], [-3.2, 1.5], [-4.0, 0.0],
    ])


def _edge_path_follows_segment(points, edges, start, end, tolerance=1e-9):
    """Return whether mesh edges form the complete straight contour segment."""

    segment = points[end] - points[start]
    length_squared = float(np.dot(segment, segment))
    offsets = points - points[start]
    cross = np.abs(segment[0] * offsets[:, 1] - segment[1] * offsets[:, 0])
    projections = offsets @ segment
    on_segment = (
        (cross <= tolerance * max(np.sqrt(length_squared), 1.0))
        & (projections >= -tolerance)
        & (projections <= length_squared + tolerance)
    )
    adjacency = {index: set() for index in np.flatnonzero(on_segment)}
    for a, b in edges:
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    pending = [start]
    visited = set()
    while pending:
        current = pending.pop()
        if current == end:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency.get(current, ()))
    return False


def test_airfoil_section_closes_at_both_edges_and_scales_with_local_chord():
    large = airfoil_max_full_thickness(4.0, 1.0)
    small = airfoil_max_full_thickness(0.25, 1.0)
    section = airfoil_full_thickness(np.linspace(0.0, 4.0, 41), 4.0, large)

    assert section[0] == 0.0
    assert section[-1] == 0.0
    assert section.max() > 0.0
    assert small <= 0.12 * 0.25
    assert small < large


def test_airfoil_reconstruction_restores_dense_structured_single_chord_mesh():
    source = _coarse_scene()
    rebuilt = rebuild_texture_scene_airfoil(
        source, span_stations=41, chord_stations=13
    )
    assert airfoil_mesh_strategy(source, station_count=41) == "structured-single-chord"
    assert len(rebuilt.vertices) == 2 * ((41 - 1) * 13 + 1)
    assert len(rebuilt.faces) == 1920
    assert not np.any(rebuilt.face_labels == "silhouette_ribbon")

    skin_count = len(rebuilt.vertices) // 2
    np.testing.assert_allclose(
        rebuilt.vertices[:skin_count, :2], rebuilt.vertices[skin_count:, :2]
    )
    np.testing.assert_allclose(
        rebuilt.vertices[:skin_count, 2], -rebuilt.vertices[skin_count:, 2]
    )
    rows = rebuilt.vertices[: skin_count - 1].reshape(40, 13, 3)
    np.testing.assert_allclose(rows[:, (0, -1), 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(rebuilt.vertices[skin_count - 1, 2], 0.0, atol=1e-12)


def test_airfoil_reconstruction_accepts_two_tissue_intervals_without_bridging_notch():
    source = _scene_from_outline([
        [0.0, 0.0], [0.0, 4.0], [-1.0, 4.0], [-1.0, 2.0],
        [-3.0, 2.0], [-3.0, 4.0], [-4.0, 4.0], [-4.0, 0.0],
    ])
    rebuilt = rebuild_texture_scene_airfoil(source)

    assert airfoil_mesh_strategy(source) == "boundary-multi-interval"
    assert len(rebuilt.faces) > 10_000
    assert not np.array_equal(rebuilt.faces, source.faces)
    positive = rebuilt.faces[rebuilt.face_labels == "positive_z_skin"]
    centroids = rebuilt.vertices[positive, :2].mean(axis=1)
    polygon = Polygon(source.outline)
    assert all(polygon.covers(Point(point)) for point in centroids)
    positive_xy = rebuilt.vertices[rebuilt.vertices[:, 2] >= 0.0, :2]
    for boundary_point in source.outline:
        distances = np.linalg.norm(positive_xy - boundary_point, axis=1)
        assert distances.min() <= 1e-10
    positive_edges = {
        tuple(sorted((int(a), int(b))))
        for face in positive
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    }
    boundary_indices = [
        int(np.argmin(np.linalg.norm(positive_xy - point, axis=1)))
        for point in source.outline
    ]
    for start, end in zip(boundary_indices, boundary_indices[1:] + boundary_indices[:1]):
        assert _edge_path_follows_segment(positive_xy, positive_edges, start, end)
    assert np.isfinite(rebuilt.vertices).all()

    preview = rebuild_texture_scene_airfoil(
        source,
        span_stations=65,
        chord_stations=13,
        boundary_min_triangles=600,
    )
    preview_positive = preview.faces[
        preview.face_labels == "positive_z_skin"
    ]
    assert 600 <= len(preview_positive) < len(positive)
    preview_xy = preview.vertices[: len(preview.vertices) // 2, :2]
    preview_edges = {
        tuple(sorted((int(a), int(b))))
        for face in preview_positive
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    }
    preview_boundary = [
        int(np.argmin(np.linalg.norm(preview_xy - point, axis=1)))
        for point in source.outline
    ]
    for start, end in zip(
        preview_boundary, preview_boundary[1:] + preview_boundary[:1]
    ):
        assert _edge_path_follows_segment(preview_xy, preview_edges, start, end)


def test_dense_multi_interval_airfoil_renders_a_smooth_leading_silhouette():
    from finshape_surfer_gui import _load_module, baseline

    leading_y = np.linspace(0.0, 5.0, 401)
    leading = np.column_stack((-2.0 * (leading_y / 5.0) ** 0.8, leading_y))
    trailing = np.array([
        [-7.0, 5.0], [-6.8, 4.0], [-5.1, 4.0], [-5.0, 3.4],
        [-6.4, 3.4], [-6.0, 2.2], [-5.7, 0.0],
    ])
    outline = np.vstack((leading, trailing))
    source = replace(_coarse_scene(), outline=outline)
    rebuilt = rebuild_texture_scene_airfoil(
        source, span_stations=161, chord_stations=21
    )
    assert airfoil_mesh_strategy(source) == "boundary-multi-interval"

    renderer = _load_module("finshape_hq_multi_interval_regression", baseline.HQ_RENDERER_PATH)
    scene = SimpleNamespace(
        vertices=rebuilt.vertices,
        faces=rebuilt.faces,
        face_labels=rebuilt.face_labels,
        textured_skin=rebuilt.textured_skin,
        vertex_uv_px=np.zeros((len(rebuilt.vertices), 2), dtype=float),
        texture_rgb=np.full((2, 2, 3), 30, dtype=np.uint8),
    )
    rendered = render_smooth_photo_frame(
        renderer, scene, 600, 500, pitch=0.0, yaw=-90.0, roll=0.0, supersample=1
    )
    foreground = np.any(np.asarray(rendered.image) < 200, axis=2)
    occupied_rows = np.flatnonzero(foreground.sum(axis=1) > 10)
    right_edge = np.asarray([
        np.max(np.flatnonzero(foreground[row])) for row in occupied_rows
    ])
    assert np.percentile(np.abs(np.diff(right_edge, n=2)), 99) <= 1.0


def test_structured_airfoil_renders_a_smooth_oblique_silhouette():
    from finshape_surfer_gui import _load_module, baseline

    parameter = np.linspace(0.0, 1.0, 401)
    leading = np.column_stack((-3.0 * parameter**0.72, 5.0 * parameter))
    trailing = np.column_stack((-8.0 + 5.0 * parameter**0.83, 5.0 * parameter))
    outline = np.vstack((leading, trailing[-2::-1]))
    source = replace(_coarse_scene(), outline=outline)
    rebuilt = rebuild_texture_scene_airfoil(source)
    renderer = _load_module("finshape_hq_render_regression", baseline.HQ_RENDERER_PATH)
    scene = SimpleNamespace(
        vertices=rebuilt.vertices,
        faces=rebuilt.faces,
        face_labels=rebuilt.face_labels,
        textured_skin=rebuilt.textured_skin,
        vertex_uv_px=np.zeros((len(rebuilt.vertices), 2), dtype=float),
        texture_rgb=np.full((2, 2, 3), 30, dtype=np.uint8),
    )

    rendered = render_smooth_photo_frame(
        renderer,
        scene,
        600,
        500,
        pitch=4.0,
        yaw=-159.0,
        roll=0.0,
        supersample=1,
    )
    foreground = np.any(np.asarray(rendered.image) < 200, axis=2)
    occupied_rows = np.flatnonzero(foreground.sum(axis=1) > 10)
    assert len(occupied_rows) > 300
    for edge_picker in (np.min, np.max):
        edge = np.asarray([
            edge_picker(np.flatnonzero(foreground[row])) for row in occupied_rows
        ])
        second_difference = np.abs(np.diff(edge, n=2))
        assert np.percentile(second_difference, 99) <= 1.0


def test_anatomical_side_mapping_is_left_positive_z_right_negative_z():
    renderer = SimpleNamespace(VERSION="test")
    proxy = _HQDisplayProxy(renderer)

    @dataclass(frozen=True)
    class TextureBase:
        photo_side: str
        textured_skin: str

    base = TextureBase(photo_side="left", textured_skin="wrong")

    @dataclass(frozen=True)
    class TextureScene:
        photo_side: str
        textured_skin: str
        base_scene: object

    scene = TextureScene("left", "wrong", base)
    assert proxy.retarget_photo_side(scene, "left").textured_skin == "positive_z_skin"
    assert proxy.retarget_photo_side(scene, "right").textured_skin == "negative_z_skin"


def test_photo_renderer_draws_only_selected_skin_at_tangent_views():
    calls = []

    class Base:
        NEUTRAL_RGB: ClassVar[dict] = {}

        @staticmethod
        def _rasterize_triangle(*args, **kwargs):
            calls.append(kwargs["textured"])

    class Renderer:
        base = Base()

        @staticmethod
        def RenderResult(**kwargs):
            return SimpleNamespace(**kwargs)

        @staticmethod
        def project_vertices(vertices, width, height, **_kwargs):
            return np.asarray(vertices[:, :2], dtype=float), np.asarray(vertices[:, 2], dtype=float)

    scene = SimpleNamespace(
        vertices=np.array([[10, 10, 1], [30, 10, 1], [10, 30, 1], [40, 40, -1]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3]], dtype=int),
        face_labels=np.array(["positive_z_skin", "negative_z_skin", "edge_wall"]),
        textured_skin="positive_z_skin",
        vertex_uv_px=np.zeros((4, 2), dtype=float),
        texture_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
    )

    result = render_smooth_photo_frame(
        Renderer(), scene, 64, 64, pitch=0.0, yaw=90.0, roll=0.0
    )

    assert isinstance(result.image, Image.Image)
    assert calls == [True]


def test_photo_mask_substitution_never_mutates_measured_cfd_curves():
    measured = {"outline": [[0, 0], [-4, 5], [-8, 0]]}
    snapshot = {
        "curves": measured,
        "image_outline": [[100, 100], [50, 20], [10, 100]],
        "photo_mask_curves": {"outline": [[0, 0], [-3, 6], [-8, 0]]},
        "photo_mask_image_outline": [[100, 100], [55, 10], [10, 100]],
    }

    adapted = photo_mask_snapshot(snapshot)

    assert adapted["curves"] == snapshot["photo_mask_curves"]
    assert adapted["image_outline"] == snapshot["photo_mask_image_outline"]
    assert snapshot["curves"] is measured
    assert snapshot["curves"]["outline"][1] == [-4, 5]
    assert adapted["photo_mask_applied"] is True


def test_photo_mask_proxy_adapts_only_the_texture_renderer_input():
    captured = []

    class Renderer:
        VERSION = "test"

        @staticmethod
        def build_texture_scene(snapshot, _exporter, side, **_kwargs):
            captured.append(snapshot)
            scene = _coarse_scene()
            return Scene(
                scene.vertices,
                scene.faces,
                scene.face_labels,
                scene.face_colors,
                scene.outline,
                "positive_z_skin" if side == "left" else "negative_z_skin",
            )

    proxy = PhotoMaskTextureProxy(Renderer())
    source = {
        "curves": {"outline": [[0, 0], [-4, 5], [-8, 0]]},
        "image_outline": [[100, 100], [50, 20], [10, 100]],
        "photo_mask_curves": {"outline": [[0, 0], [-3, 6], [-8, 0]]},
        "photo_mask_image_outline": [[100, 100], [55, 10], [10, 100]],
    }
    rebuilt = proxy.build_texture_scene(source, object(), "left")
    assert len(rebuilt.faces) == 51_200
    assert captured[0]["curves"] == source["photo_mask_curves"]
    assert source["curves"]["outline"][1] == [-4, 5]
