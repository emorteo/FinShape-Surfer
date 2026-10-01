import numpy as np
from PIL import Image

from core.vision.finfindr.detection import (
    BoundingBox,
    FinFindRAssistant,
    _astar,
    _crop_boxes_from_probabilities,
    _edge_path_from_probabilities,
    _fallback_detection_tiles,
    _fin_boxes_from_edge_probabilities,
    _merge_overlapping_boxes,
    _robust_color_boundary_probability,
    _smooth_leading_path,
)


def test_crop_box_is_scaled_bounded_and_ranked():
    probabilities = np.zeros((2, 150, 225), dtype=np.float32)
    probabilities[1, 45:86, 80:121] = 0.9
    boxes = _crop_boxes_from_probabilities(
        probabilities,
        original_size=(900, 600),
        threshold=0.45,
        target_channels=(1,),
        minimum_size=50,
    )
    assert len(boxes) == 1
    box = boxes[0]
    assert 0 <= box.x0 < box.x1 <= 900
    assert 0 <= box.y0 < box.y1 <= 600
    assert box.width > 50 and box.height > 50
    assert box.confidence > 0.8


def test_astar_follows_low_cost_corridor():
    cost = np.full((20, 20), 50.0)
    cost[2:18, 9:11] = 1.0
    path = _astar(cost, (2, 10), (17, 10))
    assert tuple(path[0]) == (2, 10)
    assert tuple(path[-1]) == (17, 10)
    assert np.all((path[:, 1] >= 9) & (path[:, 1] <= 10))


def test_overlapping_crop_regions_are_merged_for_one_review_crop():
    boxes = (
        BoundingBox(10, 20, 90, 100, 0.7),
        BoundingBox(70, 50, 160, 130, 0.6),
        BoundingBox(300, 20, 380, 100, 0.5),
    )
    merged = _merge_overlapping_boxes(boxes)
    assert merged == (
        BoundingBox(10, 20, 160, 130, 0.7),
        BoundingBox(300, 20, 380, 100, 0.5),
    )


def test_fallback_tiles_cover_image_center_and_stay_in_bounds():
    tiles = _fallback_detection_tiles((1000, 600))
    assert len(tiles) == 9
    assert any(x0 <= 500 <= x1 and y0 <= 300 <= y1 for x0, y0, x1, y1 in tiles)
    assert all(0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 600 for x0, y0, x1, y1 in tiles)


def test_crop_detector_uses_tiles_only_when_global_detection_is_empty():
    empty = np.zeros((1, 2, 150, 225), dtype=np.float32)
    detected = empty.copy()
    detected[0, 0, 45:95, 70:125] = 0.95

    class Model:
        def __init__(self):
            self.calls = 0

        def predict(self, _values):
            self.calls += 1
            return detected if self.calls == 2 else empty

    assistant = object.__new__(FinFindRAssistant)
    assistant.bundle = type("Bundle", (), {"upstream_commit": "test", "directory": None})()
    assistant.device = "cpu"
    assistant._hashes = {}
    assistant._cropper = Model()
    assistant._edge_model = None

    result = assistant.detect_fins(Image.new("RGB", (1200, 800)), minimum_size=40)

    assert result.boxes
    assert result.provenance["multiscale_fallback_tiles"] == 9


def test_color_boundary_uses_dark_tissue_and_dark_water_context():
    values = np.full((100, 140, 3), (22, 31, 38), dtype=np.uint8)
    values[20:90, 35:110] = (39, 29, 22)
    polygon = ((35, 20), (110, 20), (110, 90), (35, 90))

    boundary = _robust_color_boundary_probability(Image.fromarray(values), polygon)

    assert boundary is not None
    edge_strength = float(boundary[25:85, 33:38].mean())
    interior_strength = float(boundary[35:75, 55:90].mean())
    assert edge_strength > interior_strength + 0.25


def test_color_boundary_detects_equal_luminance_blue_water_and_gray_tissue():
    values = np.full((120, 180, 3), (18, 36, 50), dtype=np.uint8)
    values[18:108, 45:145] = (35, 35, 35)
    polygon = ((45, 18), (145, 18), (145, 108), (45, 108))

    boundary = _robust_color_boundary_probability(Image.fromarray(values), polygon)

    assert boundary is not None
    edge_strength = float(boundary[30:100, 44:47].mean())
    water_strength = float(boundary[30:100, 10:30].mean())
    tissue_strength = float(boundary[35:95, 70:120].mean())
    assert edge_strength > max(water_strength, tissue_strength) + 0.35


def test_leading_edge_smoothing_removes_pixel_staircase_and_keeps_endpoints():
    y = np.arange(0.0, 100.0)
    x = 50.0 + (np.arange(len(y)) % 2) * 2.0
    points = np.column_stack((x, y))

    smoothed = _smooth_leading_path(points)

    np.testing.assert_allclose(smoothed[0], points[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(smoothed[-1], points[-1], rtol=0, atol=1e-12)
    assert np.std(np.diff(smoothed[2:-2, 0])) < np.std(np.diff(points[:, 0])) * 0.2


def test_r_target_one_maps_to_python_channel_zero():
    probabilities = np.zeros((1, 2, 150, 225), dtype=np.float32)
    probabilities[0, 0, 35:80, 25:75] = 0.9
    probabilities[0, 1, 35:80, 150:200] = 0.95

    class Model:
        def predict(self, _values):
            return probabilities

    assistant = object.__new__(FinFindRAssistant)
    assistant.bundle = type("Bundle", (), {
        "upstream_commit": "test",
        "directory": None,
    })()
    assistant.device = "cpu"
    assistant._hashes = {}
    assistant._cropper = Model()
    assistant._edge_model = None

    result = assistant.detect_fins(Image.new("RGB", (900, 600)), minimum_size=50)
    assert len(result.boxes) == 1
    assert result.boxes[0].x1 < 500
    assert result.provenance["r_target"] == 1
    assert result.provenance["python_target_channel"] == 0


def test_edge_adapter_reaches_tip_and_trailing_base():
    height, width = 100, 160
    probabilities = np.zeros((6, height, width), dtype=np.float32)
    rows = np.arange(12, 91)
    cols = np.rint(55 + 0.45 * (rows - 12)).astype(int)
    probabilities[3, rows, cols] = 0.95
    probabilities[3] = np.maximum(probabilities[3], np.roll(probabilities[3], 1, axis=1) * 0.8)
    probabilities[4, 8:20, 50:63] = 0.95
    probabilities[5, rows, np.minimum(cols + 45, width - 1)] = 0.9
    image = Image.fromarray(np.tile(np.arange(width, dtype=np.uint8), (height, 1)))

    points, confidence = _edge_path_from_probabilities(
        image.resize((320, 300)).convert("RGB"), probabilities, threshold=0.35
    )
    assert points.shape[1] == 2
    assert points[0, 1] < 70
    assert points[-1, 1] > 240
    assert confidence > 0.5


def test_operator_landmarks_are_exact_edge_endpoints_and_preserve_notch():
    height, width = 120, 180
    probabilities = np.zeros((6, height, width), dtype=np.float32)
    rows = np.arange(10, 111)
    cols = np.where(rows < 55, 45, 65).astype(int)
    probabilities[3, rows, cols] = 0.95
    probabilities[4, 7:18, 40:51] = 0.95
    image_values = np.full((height, width, 3), 220, dtype=np.uint8)
    for row, col in zip(rows, cols):
        image_values[row, col:] = 25
    image = Image.fromarray(image_values)
    endpoints = ((45.25, 10.5), (65.75, 110.25))

    points, _confidence = _edge_path_from_probabilities(
        image, probabilities, threshold=0.35, endpoints_xy=endpoints
    )

    np.testing.assert_allclose(points[0], endpoints[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(points[-1], endpoints[1], rtol=0, atol=1e-6)
    assert points[:, 0].max() >= 64


def test_fin_refinement_excludes_long_dolphin_back_extension():
    probabilities = np.zeros((6, 100, 160), dtype=np.float32)
    probabilities[4, 18:25, 92:100] = 0.95  # tip
    for row in range(22, 66):
        probabilities[5, row, 94 - (row - 22) // 2] = 0.9  # leading edge
        probabilities[3, row, 98 + (row - 22) // 5] = 0.9  # trailing edge
    probabilities[3, 65, 20:102] = 0.9  # dolphin back, not the fin

    boxes = _fin_boxes_from_edge_probabilities(
        probabilities,
        original_size=(1600, 1000),
        coarse_boxes=(BoundingBox(100, 100, 1500, 900, 0.9),),
    )

    assert len(boxes) == 1
    assert boxes[0].x0 > 500
    assert boxes[0].x1 < 1250
    assert boxes[0].y1 < 800


def test_full_resolution_refinement_preserves_a_sharp_notch():
    height, width = 180, 240
    probabilities = np.zeros((6, 90, 120), dtype=np.float32)
    image = np.full((height, width, 3), 225, dtype=np.uint8)
    contour = []
    for y in range(20, 161):
        x = 80 if y < 75 else (112 if y < 100 else 82)
        contour.append((x, y))
        image[y, x:] = 20
        probabilities[3, y // 2, x // 2] = 0.95
    probabilities[4, 8:16, 36:45] = 0.95
    endpoints = ((80.0, 20.0), (82.0, 160.0))

    points, _confidence = _edge_path_from_probabilities(
        Image.fromarray(image), probabilities, threshold=0.35, endpoints_xy=endpoints
    )

    np.testing.assert_allclose(points[0], endpoints[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(points[-1], endpoints[1], rtol=0, atol=1e-6)
    assert len(points) >= 140
    assert points[(points[:, 1] >= 76) & (points[:, 1] <= 98), 0].max() >= 108


def test_full_photo_contour_uses_both_edge_classes_and_exact_landmark_anchors(monkeypatch):
    calls = []

    def fake_path(_image, probabilities, _threshold, endpoints_xy=None, **_kwargs):
        endpoints = np.asarray(endpoints_xy, dtype=np.float32)
        calls.append((float(probabilities[3, 0, 0]), endpoints.copy()))
        midpoint = endpoints.mean(axis=0, keepdims=True)
        return np.vstack((endpoints[0], midpoint[0], endpoints[1])), 0.8

    monkeypatch.setattr(
        "core.vision.finfindr.detection._edge_path_from_probabilities", fake_path
    )
    probabilities = np.zeros((1, 6, 4, 4), dtype=np.float32)
    probabilities[0, 3] = 0.3
    probabilities[0, 5] = 0.9

    class Model:
        def predict(self, _values):
            return probabilities

    assistant = object.__new__(FinFindRAssistant)
    assistant.bundle = type("Bundle", (), {"upstream_commit": "test", "directory": None})()
    assistant.device = "cpu"
    assistant._hashes = {}
    assistant._edge_model = Model()
    assistant._cropper = None
    result = assistant.suggest_fin_contour(
        Image.new("RGB", (120, 100)), b_xy=(100, 90), a_xy=(50, 10), c30_xy=(20, 90)
    )

    np.testing.assert_allclose([value for value, _points in calls], [0.9, 0.3])
    np.testing.assert_allclose(result.leading_points[[0, -1]], ((100, 90), (50, 10)))
    np.testing.assert_allclose(result.trailing_points[[0, -1]], ((50, 10), (20, 90)))
    np.testing.assert_allclose(result.outline_points[[0, -1]], ((100, 90), (20, 90)))
    assert result.provenance["photo_id_mask_only"] is True
    assert result.provenance["cfd_surface_changed"] is False
