from types import SimpleNamespace

import numpy as np
from shapely.geometry import Point, Polygon

from finshape_surfer_gui import (
    IMAGE_ALGORITHM_CREDIT,
    INTERACTIVE_MOTION_MAX_PIXELS,
    MORPHOMETRY_CREDIT,
    TIP_CAP_START_FRACTION,
    IntegratedFinShapeApp,
    _airfoil_tip_cap_grid,
    _anchored_te2_curve,
    _camera_drag_value,
    _clean_measurement_example,
    _draw_assisted_edge_3d,
    _fit_wireframe_axes,
    _HQDisplayProxy,
    _hybrid_photo_id_outline,
    _nearest_control_field,
    _prepare_c10_anchored_control,
    _readable_rail_fractions,
    _uniform_section_y_values,
    baseline,
    fin2d,
)


def test_wireframe_uses_odd_uniform_rail_count_with_a_centerline():
    fractions = _readable_rail_fractions(117, max_rails=35)

    assert len(fractions) == 15
    np.testing.assert_allclose(
        np.diff(fractions),
        np.full(14, 1.0 / 14.0),
        rtol=0,
        atol=1e-12,
    )
    assert fractions[len(fractions) // 2] == 0.5


def test_tip_cap_rails_end_at_unique_equally_spaced_measured_contour_points():
    fractions = np.linspace(0.0, 1.0, 9)
    rails = []
    for fraction in fractions:
        ys = np.array([0.0, 4.0, 8.0])
        xs = np.full_like(ys, 2.0 - 4.0 * fraction)
        thickness = np.full_like(ys, 0.3 * np.sin(np.pi * fraction))
        rails.append((xs, ys, thickness, -thickness))
    theta = np.linspace(0.0, np.pi, 401)
    tip = np.column_stack((2.0 * np.cos(theta), 8.0 + 2.0 * np.sin(theta)))
    outline = np.vstack(([[2.0, 0.0], [2.0, 8.0]], tip[1:], [[-2.0, 0.0]]))

    upper, lower, rings, targets = _airfoil_tip_cap_grid(
        rails, fractions, outline, samples=31
    )

    assert len(np.unique(np.round(targets, 8), axis=0)) == len(fractions)
    segment_lengths = np.linalg.norm(np.diff(targets, axis=0), axis=1)
    np.testing.assert_allclose(
        segment_lengths, np.full_like(segment_lengths, segment_lengths.mean()), rtol=2e-3
    )
    np.testing.assert_allclose(upper[:, -1, :2], targets, atol=1e-12)
    np.testing.assert_allclose(upper[:, -1, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(lower[:, :, :2], upper[:, :, :2], atol=1e-12)
    np.testing.assert_allclose(lower[:, :, 2], -upper[:, :, 2], atol=1e-12)
    center = int(np.argmax(upper[:, 0, 2]))
    center_y = upper[center, :, 1]
    center_z = upper[center, :, 2]
    parameter = np.linspace(0.0, 1.0, upper.shape[1])
    eased = 0.5 - 0.5 * np.cos(np.pi * parameter)
    np.testing.assert_allclose(
        (center_z / center_z[0]) ** 2 + eased**2,
        np.ones_like(eased),
        rtol=0.0,
        atol=1e-12,
    )
    linear_closure = center_z[0] * (
        1.0 - (center_y - center_y[0]) / (center_y[-1] - center_y[0])
    )
    assert np.all(center_z >= linear_closure - 1e-12)
    assert center_z[len(center_z) // 2] > 1.25 * linear_closure[len(center_z) // 2]
    assert len(rings) == 4
    for value, (upper_ring, lower_ring) in zip((0.20, 0.40, 0.60, 0.80), rings):
        sample_index = round(value * (upper.shape[1] - 1))
        np.testing.assert_allclose(upper_ring, upper[:, sample_index], atol=1e-12)
        np.testing.assert_allclose(lower_ring, lower[:, sample_index], atol=1e-12)

    body = upper[:, 0, :2]
    cap_polygon = Polygon(np.vstack((body, targets[::-1])))
    assert all(
        cap_polygon.buffer(1e-9).covers(Point(point))
        for point in upper[:, :, :2].reshape(-1, 2)
    )


def test_workflow_credits_identify_actual_engines_and_authors():
    assert "FinShape2D R-GUI 2.0.20" in IMAGE_ALGORITHM_CREDIT
    assert "Eduardo Morteo & Rodrigo Morteo" in IMAGE_ALGORITHM_CREDIT
    assert MORPHOMETRY_CREDIT == (
        "Ratios by David Weller & Eduardo Morteo, in: Morteo et al. (2017)"
    )


def test_wireframe_sections_are_uniform_and_keep_curve_junctions():
    curves = {
        "le0": np.array([[0.0, 0.0], [1.0, 2.13]]),
        "le1": np.array([[1.0, 2.13], [2.0, 5.0]]),
        "te1": np.array([[2.0, 5.0], [3.0, 3.27]]),
        "te2": np.array([[3.0, 3.27], [4.0, 1.41]]),
        "te3": np.array([[4.0, 1.41], [5.0, 0.0]]),
    }

    values = _uniform_section_y_values(curves, 5.0, 39)

    for anchor in (0.0, 1.41, 2.13, 3.27):
        assert np.any(np.isclose(values, anchor, rtol=0, atol=1e-9))
    assert np.isclose(values.max(), TIP_CAP_START_FRACTION * 5.0)
    assert len(values) <= 43


def test_rendered_wireframe_has_a_meshed_tip_cap_without_a_single_apex_pole():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    y = np.linspace(0.0, 5.0, 401)
    leading = np.column_stack((-7.0 + 0.8 * (y / 5.0) ** 1.4, y))
    trailing = np.column_stack((-10.0 + 3.8 * (y / 5.0) ** 0.72, y))
    outline = np.vstack((leading, trailing[-2::-1]))
    curves = {"leading": leading, "trailing": trailing, "outline": outline}

    app = object.__new__(IntegratedFinShapeApp)
    app._fin3d_module = None
    module = IntegratedFinShapeApp._load_fin3d(app)
    module._surfer_app_ref = lambda: None
    figure = Figure(figsize=(7.2, 5.4), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111, projection="3d")
    module.draw_3d_wireframe_axes(
        axis, curves, "rendered-tip-regression", pitch=18.0, yaw=-59.0, roll=4.0
    )
    canvas.draw()

    cap_endpoints = []
    for line in axis.lines:
        if line.get_color() != "blue":
            continue
        xs, zs, ys = (np.asarray(values) for values in line.get_data_3d())
        if len(xs) == 25 and ys[-1] > TIP_CAP_START_FRACTION * 5.0:
            cap_endpoints.append((xs[-1], ys[-1], zs[-1]))
        assert not (
            np.ptp(ys) > 1e-9
            and np.ptp(zs) <= 1e-9
            and abs(float(zs[0])) > 1e-9
        )
    unique_endpoints = np.unique(np.round(cap_endpoints, 7), axis=0)
    assert len(unique_endpoints) >= 13
    assert np.ptp(unique_endpoints[:, 0]) > 0.2
    assert np.ptp(unique_endpoints[:, 1]) > 0.2


def test_wireframe_airfoil_tapers_to_zero_without_tip_or_edge_floors():
    app = object.__new__(IntegratedFinShapeApp)
    app._fin3d_module = None
    module = IntegratedFinShapeApp._load_fin3d(app)
    module._surfer_app_ref = lambda: None

    maxima = []
    for chord in (4.0, 1.0, 0.25, 0.0):
        shoulder, maximum = module.morteo_SP_T(chord)
        section = module.structural_visual_thickness(
            np.linspace(0.0, chord, 101),
            chord,
            shoulder,
            maximum,
            edge_min_thickness=99.0,
        )
        maxima.append(float(section.max(initial=0.0)))
        assert section[0] == 0.0
        assert section[-1] == 0.0
        assert maximum <= 0.12 * chord + 1e-12

    assert maxima[0] > maxima[1] > maxima[2] > maxima[3] == 0.0


def test_surface_snapshot_keeps_display_airfoil_out_of_cfd_export(monkeypatch):
    monkeypatch.setattr(
        baseline.IntegratedFinShapeApp,
        "_surface_export_snapshot",
        lambda _self, _curves, _fin_id: {"provenance": {}},
    )
    app = object.__new__(IntegratedFinShapeApp)
    app._assisted_crop_record = None
    app._assisted_edge_record = None

    snapshot = IntegratedFinShapeApp._surface_export_snapshot(app, {}, "test-fin")
    display = snapshot["provenance"]["display_geometry"]

    assert display["cfd_surface_changed"] is False
    assert display["cfd_boundary_source"].startswith("measured Bezier contour")
    assert "not solver-export geometry" in display["cfd_export_decision"]


def test_vision_job_uses_explicit_active_button_after_action_rename(monkeypatch):
    import finshape_surfer_gui as surfer_gui

    started = []

    class DeferredThread:
        def __init__(self, *, target, name, daemon):
            started.append((target, name, daemon))

        def start(self):
            pass

    monkeypatch.setattr(surfer_gui.threading, "Thread", DeferredThread)
    active = []
    app = SimpleNamespace(
        _vision_busy=False,
        _vision_generation=0,
        _set_active_button=active.append,
        _update_button_states=lambda: None,
        status=SimpleNamespace(set=lambda _message: None),
    )

    IntegratedFinShapeApp._run_vision_job(
        app,
        "Automatic edge detection",
        object(),
        lambda _assistant, _image: None,
        lambda _result: None,
        active_button="suggest_edge",
    )

    assert active == ["suggest_edge"]
    assert started[0][1:] == ("finfindr-assistance", True)


def test_landmark_sequence_starts_with_a_then_b():
    app = object.__new__(IntegratedFinShapeApp)
    app.image_path = object()
    app.locked_after_save = False
    app.controls = None
    app.mode = "lm_A"
    app.landmarks = {}
    app._event_canvas_xy = lambda _event: (10.0, 20.0)
    app._canvas_to_working = lambda x, y: (x, y)
    app._confirm_landmark_click = lambda _point: True
    messages = []
    app.status = SimpleNamespace(set=messages.append)
    app._render_image = lambda: None

    IntegratedFinShapeApp.on_mouse_down(app, SimpleNamespace())
    assert app.landmarks == {"A": (10.0, 20.0)}
    assert app.mode == "lm_B"

    app._event_canvas_xy = lambda _event: (80.0, 90.0)
    IntegratedFinShapeApp.on_mouse_down(app, SimpleNamespace())
    assert app.landmarks["B"] == (80.0, 90.0)
    assert app.mode == "lm_C30"
    assert messages == ["Landmarks: click B.", "Landmarks: click C30."]


def test_measurement_example_cleanup_removes_only_top_right_artifact():
    image = fin2d.Image.new("RGB", (100, 100), "gray")
    image.putpixel((10, 10), (1, 2, 3))

    cleaned = _clean_measurement_example(image)

    assert cleaned.getpixel((90, 5)) == (255, 255, 255)
    assert cleaned.getpixel((10, 10)) == (1, 2, 3)


def test_b_to_d_cubic_exposes_independent_b_and_d_control_handles():
    landmarks = fin2d.Landmarks(
        A=(0.0, 0.0), B=(10.0, 10.0), C5=(1.0, 2.0), C10=(2.0, 4.0),
        C20=(3.0, 6.0), C30=(4.0, 8.0), D=(5.0, 5.0),
    )
    controls = fin2d.build_initial_controls(landmarks)

    assert "LE0_P1" in fin2d.DRAGGABLE_BEZIER_CONTROL_FIELDS
    assert "LE0_P2" in fin2d.DRAGGABLE_BEZIER_CONTROL_FIELDS
    assert controls.LE0_P1 != controls.LE0_P2


def test_camera_drag_right_decreases_reversed_yaw_and_clamps_limits():
    assert _camera_drag_value(20.0, 10.0, -180.0, 180.0) == 16.5
    assert _camera_drag_value(-175.0, 100.0, -180.0, 180.0) == -180.0


def test_clearing_vision_suggestions_does_not_change_manual_measurement_state():
    deleted_tags = []
    landmarks = {"A": (10.0, 20.0), "B": (30.0, 40.0)}
    controls = object()
    app = SimpleNamespace(
        _vision_generation=7,
        _suggested_fin_box=(1, 2, 3, 4),
        _suggested_trailing_edge=[(1.0, 2.0)],
        landmarks=landmarks,
        controls=controls,
        crop_box=(5, 6, 70, 80),
        canvas=SimpleNamespace(delete=lambda *tags: deleted_tags.extend(tags)),
        btn_apply_detected_crop=None,
    )

    IntegratedFinShapeApp._invalidate_vision_suggestions(app)

    assert app._vision_generation == 8
    assert app._suggested_fin_box is None
    assert app._suggested_trailing_edge is None
    assert app.landmarks is landmarks
    assert app.controls is controls
    assert app.crop_box == (5, 6, 70, 80)
    assert deleted_tags == ["assisted_crop", "assisted_edge"]


def test_active_launcher_extends_frozen_release_without_replacing_it():
    assert IntegratedFinShapeApp.__mro__[1].__module__ == "finshape_surfer_gui_1_2_2_frozen"


def test_nearest_control_handle_wins_even_when_fixed_landmark_is_drawn_above_it():
    controls = SimpleNamespace(TE2_Q1=(10.0, 10.0), TE1_P1=(30.0, 30.0))
    selected = _nearest_control_field(
        controls,
        ("TE2_Q1", "TE1_P1"),
        working_point=(10.5, 10.25),
        display_scale=2.0,
        max_distance_px=11.0,
    )
    assert selected == "TE2_Q1"


def test_control_handle_outside_hit_radius_is_not_selected():
    controls = SimpleNamespace(TE2_Q1=(10.0, 10.0))
    assert _nearest_control_field(
        controls,
        ("TE2_Q1",),
        working_point=(30.0, 30.0),
        display_scale=1.0,
        max_distance_px=11.0,
    ) is None


def test_te2_control_is_unrestricted_and_piecewise_curve_stays_on_c10():
    p0 = (0.0, 0.0)
    c10 = (4.0, 3.0)
    p2 = (10.0, 0.0)
    free_control = (-12.0, 18.0)
    curve = _anchored_te2_curve(p0, free_control, c10, p2, 81)

    np.testing.assert_allclose(curve[40], c10, rtol=0, atol=1e-12)
    assert curve[:, 0].min() < p0[0]


def test_active_control_curve_uses_c10_marker_without_changing_frozen_controls():
    landmarks = fin2d.Landmarks(
        A=(0.0, 0.0), B=(10.0, 10.0), C5=(1.0, 2.0), C10=(2.0, 4.0),
        C20=(3.0, 6.0), C30=(4.0, 8.0), D=(5.0, 5.0),
    )
    controls = fin2d.build_initial_controls(landmarks)
    frozen_by_count = {
        count: np.asarray(fin2d.curves_from_controls(controls, n_per_seg=count)["TE2"])
        for count in (80, 81)
    }
    _prepare_c10_anchored_control(controls, landmarks.C10)
    for count, frozen in frozen_by_count.items():
        active = np.asarray(fin2d.curves_from_controls(controls, n_per_seg=count)["TE2"])
        np.testing.assert_allclose(active, frozen, rtol=0, atol=1e-12)

    frozen = frozen_by_count[81]
    active = np.asarray(fin2d.curves_from_controls(controls, n_per_seg=81)["TE2"])
    np.testing.assert_allclose(active[40], landmarks.C10, rtol=0, atol=1e-12)

    controls.TE2_Q1 = (-20.0, 18.0)
    moved = np.asarray(fin2d.curves_from_controls(controls, n_per_seg=81)["TE2"])
    assert moved[:, 0].min() < frozen[:, 0].min()
    np.testing.assert_allclose(moved[40], landmarks.C10, rtol=0, atol=1e-12)


def test_hybrid_photo_outline_uses_smooth_bezier_leading_and_exact_trailing_detail():
    landmarks = fin2d.Landmarks(
        A=(0.0, 0.0), B=(10.0, 10.0), C5=(-1.0, 2.0), C10=(-2.0, 4.0),
        C20=(-3.0, 6.0), C30=(-4.0, 8.0), D=(5.0, 5.0),
    )
    controls = fin2d.build_initial_controls(landmarks)
    trailing = np.array([[0.0, 0.0], [-0.5, 1.0], [-0.2, 1.5], [-4.0, 8.0]])

    outline = _hybrid_photo_id_outline(controls, trailing, n_per_leading_segment=21)
    leading = fin2d.curves_from_controls(controls, n_per_seg=21)["LE0"]
    leading += fin2d.curves_from_controls(controls, n_per_seg=21)["LE1"][1:]

    np.testing.assert_allclose(outline[:len(leading)], leading)
    np.testing.assert_allclose(outline[len(leading):], trailing[1:])


def test_saved_texture_snapshot_can_receive_later_photo_mask_without_mutation():
    landmarks = fin2d.Landmarks(
        A=(0.0, 0.0), B=(10.0, 10.0), C5=(-1.0, 2.0), C10=(-2.0, 4.0),
        C20=(-3.0, 6.0), C30=(-4.0, 8.0), D=(5.0, 5.0),
    )
    controls = fin2d.build_initial_controls(landmarks)
    source = {
        "standardized_landmarks": {"B": (0.0, 0.0), "A": (-4.0, 5.0), "C30": (-8.0, 0.0)},
    }
    app = object.__new__(IntegratedFinShapeApp)
    app.controls = controls
    app.landmarks = {name: getattr(landmarks, name) for name in ("A", "B", "C30")}
    app._suggested_trailing_edge = np.array([
        landmarks.A, (-0.5, 1.0), (-2.0, 5.0), landmarks.C30,
    ])
    app._assisted_edge_record = {"used_as_photo_id_mask": True}

    adapted = IntegratedFinShapeApp._apply_photo_mask_to_snapshot(app, source)

    assert "photo_mask_curves" not in source
    assert adapted["photo_mask_scope"] == "high-resolution Photo-ID rendering only"
    np.testing.assert_allclose(adapted["photo_mask_curves"]["outline"][0], (0.0, 0.0))
    np.testing.assert_allclose(adapted["photo_mask_curves"]["outline"][-1], (-8.0, 0.0))


def test_interactive_photo_preview_is_bounded_but_export_keeps_full_dimensions(monkeypatch):
    from dataclasses import dataclass

    from PIL import Image

    import finshape_surfer_gui as surfer_gui

    @dataclass(frozen=True)
    class Result:
        image: object
        render_seconds: float
        textured_pixels: int
        output_pixels: int
        supersample: int

    calls = []

    def render(_renderer, scene, width, height, **kwargs):
        calls.append((scene, width, height, kwargs["supersample"]))
        return Result(
            image=Image.new("RGB", (width, height), "white"),
            render_seconds=0.01,
            textured_pixels=100,
            output_pixels=width * height,
            supersample=kwargs["supersample"],
        )

    monkeypatch.setattr(surfer_gui, "render_smooth_photo_frame", render)
    def project_vertices(vertices, _width, _height, **_kwargs):
        points = np.asarray(vertices, dtype=float)
        return points[:, :2] * 20 + 64, points[:, 2]

    proxy = _HQDisplayProxy(SimpleNamespace(
        VERSION="test", project_vertices=project_vertices
    ))
    full_scene = object()
    motion_scene = object()
    proxy._motion_scenes[id(full_scene)] = motion_scene

    preview = proxy.render_hq_frame(
        full_scene, 1400, 1000, pitch=0, yaw=0, roll=0, supersample=2
    )
    proxy.set_interactive_motion(True)
    motion = proxy.render_hq_frame(
        full_scene, 1400, 1000, pitch=0, yaw=0, roll=0, supersample=2
    )
    proxy.set_interactive_motion(False)
    export = proxy.render_hq_frame(
        full_scene, 1920, 1440, pitch=0, yaw=0, roll=0, supersample=2
    )

    assert calls[0][0] is full_scene
    assert calls[0][1] * calls[0][2] <= surfer_gui.INTERACTIVE_PHOTO_MAX_PIXELS
    assert preview.image.size == (1400, 1000)
    assert preview.supersample == 2
    assert calls[1][0] is motion_scene
    assert calls[1][1] * calls[1][2] <= INTERACTIVE_MOTION_MAX_PIXELS
    assert calls[1][3] == 2
    assert motion.image.size == (1400, 1000)
    assert calls[2] == (full_scene, 1920, 1440, 2)
    assert export.image.size == (1920, 1440)


def test_instruction_text_correction_preserves_embedded_measurement_example():
    class Info:
        embedded_image = object()

        def __init__(self):
            self.deleted = []
            self.inserted = []

        def configure(self, **_kwargs):
            pass

        def search(self, _text, _start, stopindex=None):
            assert stopindex == "end"
            return "12.4"

        def delete(self, start, end):
            self.deleted.append((start, end))

        def insert(self, start, text):
            self.inserted.append((start, text))

    app = object.__new__(IntegratedFinShapeApp)
    app.info = Info()
    image = app.info.embedded_image

    IntegratedFinShapeApp._correct_instruction_landmark_order(app)

    assert app.info.embedded_image is image
    old = "Landmarks: B → A → C30 → C20 → C10 → C5 → D"
    assert app.info.deleted == [("12.4", f"12.4+{len(old)}c")]
    assert app.info.inserted == [
        ("12.4", "Landmarks: A → B → C30 → C20 → C10 → C5 → D")
    ]


def test_extension_overlays_are_replaced_before_each_drag_frame(monkeypatch):
    calls = []

    class Canvas:
        def delete(self, *tags):
            calls.append(("delete", tags))

        def create_line(self, *_args, **kwargs):
            calls.append(("line", kwargs.get("tags")))

        def create_rectangle(self, *_args, **kwargs):
            calls.append(("rectangle", kwargs.get("tags")))

        def tag_lower(self, *_args):
            pass

    monkeypatch.setattr(
        baseline.IntegratedFinShapeApp,
        "_draw_overlays",
        lambda _self: calls.append(("inherited", None)),
    )
    app = object.__new__(IntegratedFinShapeApp)
    app.canvas = Canvas()
    app.display_scale = 1.0
    app.crop_box = None
    app._suggested_fin_box = None
    app._suggested_photo_contour = None
    app.landmarks = {"B": (0.0, 0.0), "D": (5.0, 5.0)}
    app.controls = SimpleNamespace(LE0_P1=(1.0, 1.0), LE0_P2=(4.0, 4.0))

    IntegratedFinShapeApp._draw_overlays(app)
    IntegratedFinShapeApp._draw_overlays(app)

    deletes = [call for call in calls if call[0] == "delete"]
    assert deletes == [
        ("delete", ("assisted_crop", "assisted_edge", "leading_control_link")),
        ("delete", ("assisted_crop", "assisted_edge", "leading_control_link")),
    ]


def test_manual_crop_release_does_not_render_the_cropped_image_early(monkeypatch):
    app = object.__new__(IntegratedFinShapeApp)
    app.crop_box = None
    app._vision_generation = 0
    app._suggested_fin_box = None
    app._suggested_trailing_edge = None
    app._crop_preview_pending = False
    app.btn_apply_detected_crop = None
    app.canvas = SimpleNamespace(delete=lambda *_tags: None)
    rendered = []
    app._render_image = lambda: rendered.append(True)

    def inherited_mouse_up(self, _event):
        self.crop_box = (10, 20, 100, 120)

    monkeypatch.setattr(baseline.IntegratedFinShapeApp, "on_mouse_up", inherited_mouse_up)
    IntegratedFinShapeApp.on_mouse_up(app, SimpleNamespace())

    assert app.crop_box == (10, 20, 100, 120)
    assert app._crop_preview_pending is True
    assert rendered == []


def test_pending_manual_crop_stays_non_destructive_across_redraws(monkeypatch):
    observed = []
    app = object.__new__(IntegratedFinShapeApp)
    app._crop_preview_pending = True

    def inherited_transforms(self, exclude_crop=False):
        observed.append(exclude_crop)
        return "image"

    monkeypatch.setattr(
        baseline.IntegratedFinShapeApp, "_apply_transforms", inherited_transforms
    )

    assert IntegratedFinShapeApp._apply_transforms(app) == "image"
    assert observed == [True]


def test_apply_detected_crop_removes_green_review_rectangle():
    deleted = []
    applied = []
    app = object.__new__(IntegratedFinShapeApp)
    app._suggested_fin_box = (10, 20, 100, 120)
    app._assisted_crop_record = {"accepted": False}
    app.locked_after_save = False
    app.canvas = SimpleNamespace(delete=lambda *tags: deleted.extend(tags))
    app.apply_crop = lambda: applied.append(True)
    app.status = SimpleNamespace(set=lambda _value: None)
    app._update_button_states = lambda: None

    IntegratedFinShapeApp.apply_detected_crop(app)

    assert app.crop_box == (10, 20, 100, 120)
    assert app._suggested_fin_box is None
    assert app._assisted_crop_record["accepted"] is True
    assert deleted == ["assisted_crop", "assisted_crop"]
    assert applied == [True]


def test_saved_measurement_keeps_read_only_edge_detection_available(monkeypatch):
    states = {}

    class Button:
        def __init__(self, name):
            self.name = name

        def configure(self, **kwargs):
            if "state" in kwargs:
                states[self.name] = kwargs["state"]

    monkeypatch.setattr(
        baseline.IntegratedFinShapeApp, "_update_button_states", lambda _self: None
    )
    app = object.__new__(IntegratedFinShapeApp)
    app.image_path = object()
    app.locked_after_save = True
    app._vision_busy = False
    app._suggested_fin_box = None
    app.landmarks = {"A": (0, 1), "B": (1, 0), "C30": (-1, 0)}
    app.btn_detect_fin = Button("detect")
    app.btn_apply_detected_crop = Button("crop")
    app.btn_suggest_edge = Button("edge")

    IntegratedFinShapeApp._update_button_states(app)

    assert states == {"detect": "disabled", "crop": "disabled", "edge": "normal"}


def test_closing_2d_review_window_restores_main_review_actions():
    calls = []

    class Window:
        def winfo_exists(self):
            return True

        def destroy(self):
            calls.append("destroy")

    app = object.__new__(IntegratedFinShapeApp)
    app._set_active_button = lambda value: calls.append(("active", value))
    app._update_button_states = lambda: calls.append("states")
    app.lift = lambda: calls.append("lift")
    app.focus_set = lambda: calls.append("focus")

    IntegratedFinShapeApp._close_2d_review_window(app, Window())

    assert calls == ["destroy", ("active", None), "states", "lift", "focus"]


def test_suggested_edge_is_drawn_on_wireframe_without_replacing_geometry():
    plotted = []
    labels = []
    axis = SimpleNamespace(
        transAxes=object(),
        plot=lambda *args, **kwargs: plotted.append((args, kwargs)),
        text2D=lambda *args, **kwargs: labels.append((args, kwargs)),
    )
    edge = np.array([[-8.0, 0.0], [-7.5, 2.0], [-8.3, 5.0]])

    assert _draw_assisted_edge_3d(axis, edge) is True
    assert len(plotted) == 1
    np.testing.assert_allclose(plotted[0][0][0], edge[:, 0])
    np.testing.assert_allclose(plotted[0][0][1], 0.0)
    np.testing.assert_allclose(plotted[0][0][2], edge[:, 1])
    assert plotted[0][1]["color"] == "#ff8c00"
    assert "measured wireframe unchanged" in labels[0][0][2]


def test_rotate_control_positive_value_is_clockwise_internal_rotation():
    app = object.__new__(IntegratedFinShapeApp)
    app.rotate_deg = 0.0
    app.rot_var = SimpleNamespace(get=lambda: 12.5)
    app.image_path = object()
    app.locked_after_save = False
    app._invalidate_vision_suggestions = lambda **_kwargs: None
    app._disable_2d = lambda: None
    fitted = []
    app._auto_fit_image = lambda: fitted.append(True)

    IntegratedFinShapeApp.on_rotate_change(app)

    assert app.rotate_deg == -12.5
    assert fitted == [True]


def test_wireframe_labels_never_exceed_title_and_axes_fill_canvas():
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    axis.set_title("title", fontsize=9)
    axis.set_xlabel("X", fontsize=42)
    axis.set_ylabel("Z", fontsize=42)
    axis.set_zlabel("Y", fontsize=42)
    axis.set_xlim(-10, 1)
    axis.set_ylim(-1, 1)
    axis.set_zlim(0, 5)

    _fit_wireframe_axes(axis)

    title_size = axis.title.get_fontsize()
    assert axis.xaxis.label.get_fontsize() <= title_size
    assert axis.yaxis.label.get_fontsize() <= title_size
    assert axis.zaxis.label.get_fontsize() <= title_size
    assert axis.get_position(original=True).width == 1.0
    plt.close(figure)
