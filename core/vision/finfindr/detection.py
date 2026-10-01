"""Fin and trailing-edge suggestions based on the pinned FinFindR models.

The neural checkpoints are executed as published. The Python path extraction
is an explicit FinShape Surfer adaptation because the FinFindR R routine mixes
model inference, image cleanup, endpoint heuristics, and compiled A* helpers.
Every result therefore requires manual review before it can affect a
measurement.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .assets import ModelBundle, locate_model_bundle, model_hashes
from .errors import FinFindRModelError
from .executor import MXNetSymbolExecutor


@dataclass(frozen=True)
class BoundingBox:
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class FinDetectionResult:
    boxes: Tuple[BoundingBox, ...]
    probability_map: np.ndarray
    threshold: float
    provenance: Dict[str, object]
    requires_manual_review: bool = True


@dataclass(frozen=True)
class TrailingEdgeResult:
    points: np.ndarray
    confidence: float
    trailing_probability: np.ndarray
    tip_probability: np.ndarray
    provenance: Dict[str, object]
    requires_manual_review: bool = True


@dataclass(frozen=True)
class FinContourResult:
    """Operator-anchored learned contour for Photo-ID display only."""

    leading_points: np.ndarray
    trailing_points: np.ndarray
    outline_points: np.ndarray
    leading_confidence: float
    trailing_confidence: float
    provenance: Dict[str, object]
    requires_manual_review: bool = True


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _nchw(image: Image.Image) -> np.ndarray:
    return np.transpose(_rgb_array(image), (2, 0, 1))[None, ...]


def _resize_dominant(image: Image.Image, dominant: int = 200) -> Tuple[Image.Image, float, float]:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    scale = float(dominant) / max(width, height)
    new_size = (max(2, round(width * scale)), max(2, round(height * scale)))
    resized = image.convert("RGB").resize(new_size, Image.Resampling.BICUBIC)
    return resized, width / new_size[0], height / new_size[1]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _weighted_centroid(values: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        index = int(np.nanargmax(values))
        row, col = np.unravel_index(index, values.shape)
        return float(row), float(col)
    weights = np.maximum(values[rows, cols].astype(np.float64), 1e-8)
    return float(np.average(rows, weights=weights)), float(np.average(cols, weights=weights))


def _resize_probability(values: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BICUBIC), dtype=np.float32)


def _fallback_detection_tiles(size: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], ...]:
    """Cover an image with overlapping half-size tiles for small-animal fallback."""

    width, height = map(int, size)
    if width < 4 or height < 4:
        return ((0, 0, width, height),)
    tile_width = max(2, int(round(width * 0.58)))
    tile_height = max(2, int(round(height * 0.58)))
    x_positions = sorted({0, (width - tile_width) // 2, width - tile_width})
    y_positions = sorted({0, (height - tile_height) // 2, height - tile_height})
    return tuple(
        (x, y, min(width, x + tile_width), min(height, y + tile_height))
        for y in y_positions
        for x in x_positions
    )


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 1] to CIE L*a*b* under D65."""

    values = np.asarray(rgb, dtype=np.float32)
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear @ np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.dstack((
        116.0 * transformed[:, :, 1] - 16.0,
        500.0 * (transformed[:, :, 0] - transformed[:, :, 1]),
        200.0 * (transformed[:, :, 1] - transformed[:, :, 2]),
    ))


def _offset_box(box: BoundingBox, x_offset: int, y_offset: int) -> BoundingBox:
    return BoundingBox(
        box.x0 + x_offset,
        box.y0 + y_offset,
        box.x1 + x_offset,
        box.y1 + y_offset,
        box.confidence,
    )


def _robust_color_boundary_probability(
    image: Image.Image,
    tissue_polygon_xy: Optional[Sequence[Sequence[float]]],
) -> Optional[np.ndarray]:
    """Estimate tissue/water transitions from operator-provided interior context.

    The model is intentionally local and advisory. Eroded polygon pixels seed
    likely tissue; distant outer-frame pixels seed likely water. It does not
    classify biology or modify measurement geometry.
    """

    if tissue_polygon_xy is None:
        return None
    polygon = np.asarray(tissue_polygon_xy, dtype=float)
    width, height = image.size
    if (
        polygon.ndim != 2
        or polygon.shape[1:] != (2,)
        or len(polygon) < 3
        or not np.isfinite(polygon).all()
    ):
        raise ValueError("The tissue-reference polygon must contain finite (x, y) points.")

    mask_image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_image).polygon(
        [tuple(map(float, point)) for point in polygon], fill=255
    )
    mask = np.asarray(mask_image, dtype=np.uint8) > 0
    radius = max(2, min(18, int(round(0.012 * max(width, height)))))
    tissue_seed = ndimage.binary_erosion(mask, iterations=radius)

    border_width = max(3, int(round(0.08 * min(width, height))))
    border = np.zeros((height, width), dtype=bool)
    border[:border_width] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    # The lower frame can contain the dolphin's back, so use it only as a
    # fallback instead of teaching the water model from it immediately.
    distant = ~ndimage.binary_dilation(mask, iterations=radius)
    water_seed = border & distant
    if np.count_nonzero(water_seed) < 64:
        border[-border_width:] = True
        water_seed = border & distant
    if np.count_nonzero(tissue_seed) < 64 or np.count_nonzero(water_seed) < 64:
        return None

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    lab = _rgb_to_lab(rgb)
    total = np.maximum(rgb.sum(axis=2, keepdims=True), 1e-4)
    chroma = rgb / total
    features = np.dstack((
        lab[:, :, 0] / 100.0,
        lab[:, :, 1] / 128.0,
        lab[:, :, 2] / 128.0,
        chroma,
    ))

    def robust_distance(seed):
        samples = features[seed]
        if len(samples) > 30_000:
            step = max(1, len(samples) // 30_000)
            samples = samples[::step]
        center = np.median(samples, axis=0)
        scale = 1.4826 * np.median(np.abs(samples - center), axis=0)
        scale = np.maximum(
            scale,
            np.array([0.025, 0.018, 0.018, 0.02, 0.02, 0.02]),
        )
        return np.mean(((features - center) / scale) ** 2, axis=2)

    tissue_distance = robust_distance(tissue_seed)
    water_distance = robust_distance(water_seed)
    signed = np.clip(water_distance - tissue_distance, -20.0, 20.0)
    tissue_probability = 1.0 / (1.0 + np.exp(-0.45 * signed))
    class_boundary = np.hypot(
        ndimage.sobel(tissue_probability, axis=0),
        ndimage.sobel(tissue_probability, axis=1),
    )
    class_scale = max(float(np.quantile(class_boundary, 0.98)), 1e-6)
    class_boundary = np.clip(class_boundary / class_scale, 0.0, 1.0)

    smooth_lab = np.dstack([
        ndimage.gaussian_filter(lab[:, :, channel], sigma=0.8)
        for channel in range(3)
    ])
    local_boundary = np.sqrt(sum(
        ndimage.sobel(smooth_lab[:, :, channel], axis=0) ** 2
        + ndimage.sobel(smooth_lab[:, :, channel], axis=1) ** 2
        for channel in range(3)
    ))
    local_scale = max(float(np.quantile(local_boundary, 0.98)), 1e-6)
    local_boundary = np.clip(local_boundary / local_scale, 0.0, 1.0)
    return np.clip(
        0.80 * class_boundary + 0.20 * local_boundary, 0.0, 1.0
    ).astype(np.float32)


def _smooth_leading_path(points_xy: np.ndarray, spacing_px: float = 1.5) -> np.ndarray:
    """Remove one-pixel staircase geometry while preserving exact endpoints."""

    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 5:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = np.r_[True, np.diff(cumulative) > 1e-9]
    points, cumulative = points[keep], cumulative[keep]
    if len(points) < 5 or cumulative[-1] <= 1e-9:
        return points.copy()
    sample_count = max(5, int(math.ceil(cumulative[-1] / max(spacing_px, 0.5))) + 1)
    distance = np.linspace(0.0, cumulative[-1], sample_count)
    sampled = np.column_stack([
        np.interp(distance, cumulative, points[:, 0]),
        np.interp(distance, cumulative, points[:, 1]),
    ])
    sigma = min(3.0, max(1.0, sample_count / 160.0))
    smoothed = np.column_stack([
        ndimage.gaussian_filter1d(sampled[:, 0], sigma=sigma, mode="nearest"),
        ndimage.gaussian_filter1d(sampled[:, 1], sigma=sigma, mode="nearest"),
    ])
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def _astar(cost: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> np.ndarray:
    """Return an eight-connected minimum-cost path as row/column coordinates."""

    height, width = cost.shape
    for point in (start, end):
        if not (0 <= point[0] < height and 0 <= point[1] < width):
            raise ValueError(f"Path endpoint is outside the cost image: {point}.")
    distances = np.full((height, width), np.inf, dtype=np.float64)
    distances[start] = 0.0
    previous: Dict[Tuple[int, int], Tuple[int, int]] = {}
    queue = [(math.hypot(end[0] - start[0], end[1] - start[1]), 0.0, start)]
    steps = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    while queue:
        _estimate, distance, current = heapq.heappop(queue)
        if distance != distances[current]:
            continue
        if current == end:
            break
        for dr, dc in steps:
            row, col = current[0] + dr, current[1] + dc
            if not (0 <= row < height and 0 <= col < width):
                continue
            step_length = math.sqrt(2.0) if dr and dc else 1.0
            edge_cost = step_length * 0.5 * (float(cost[current]) + float(cost[row, col]))
            candidate = distance + edge_cost
            if candidate < distances[row, col]:
                distances[row, col] = candidate
                previous[(row, col)] = current
                heuristic = math.hypot(end[0] - row, end[1] - col) * float(np.min(cost))
                heapq.heappush(queue, (candidate + heuristic, candidate, (row, col)))

    if not np.isfinite(distances[end]):
        raise FinFindRModelError("No connected trailing-edge path could be found.")
    path = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return np.asarray(path, dtype=np.int32)


def _crop_boxes_from_probabilities(
    probabilities: np.ndarray,
    original_size: Tuple[int, int],
    threshold: float,
    target_channels: Sequence[int],
    minimum_size: int,
) -> Tuple[BoundingBox, ...]:
    if probabilities.ndim != 3:
        raise FinFindRModelError(f"Expected crop probabilities CxHxW, got {probabilities.shape}.")
    if not target_channels or any(index < 0 or index >= probabilities.shape[0] for index in target_channels):
        raise ValueError("Crop target channel is outside the model output.")

    target = np.max(probabilities[list(target_channels)], axis=0)
    target = ndimage.gaussian_filter(target, sigma=0.5)
    weak = target > (threshold / 2.0)
    if len(target_channels) > 1:
        blobs = ndimage.binary_erosion(
            ndimage.binary_dilation(weak, structure=np.ones((3, 3))),
            structure=np.ones((3, 3)),
        )
    else:
        other = np.max(np.delete(probabilities, target_channels[0], axis=0), axis=0)
        connection = ndimage.binary_dilation(weak, structure=np.ones((9, 9)))
        tightened = ndimage.binary_erosion(weak, structure=np.ones((3, 3)))
        overlap = tightened | (
            connection
            & ndimage.binary_erosion(other > threshold / 2.0, structure=np.ones((3, 3)))
        )
        overlap_labels, _ = ndimage.label(overlap, structure=np.ones((3, 3)))
        blobs = overlap_labels * tightened

    labels, count = ndimage.label(blobs > 0, structure=np.ones((3, 3)))
    strong = target > threshold
    image_width, image_height = original_size
    map_height, map_width = target.shape
    stretch_x = math.ceil(image_width / map_width)
    stretch_y = math.ceil(image_height / map_height)
    boxes = []
    for label_value in range(1, count + 1):
        component = labels == label_value
        if np.count_nonzero(component & strong) <= 1:
            continue
        rows, cols = np.nonzero(component)
        width = int(cols.max() - cols.min())
        height = int(rows.max() - rows.min())
        if min(width, height) <= 0 or max(width, height) >= 3 * min(width, height):
            continue
        margin_x, margin_y = math.ceil(width / 3), math.ceil(height / 3)
        x0 = max(0, stretch_x * (int(cols.min()) - margin_x))
        y0 = max(0, stretch_y * (int(rows.min()) - margin_y))
        x1 = min(image_width, stretch_x * (int(cols.max()) + margin_x + 1))
        y1 = min(image_height, stretch_y * (int(rows.max()) + margin_y + 1))
        if x1 - x0 <= minimum_size or y1 - y0 <= minimum_size:
            continue
        boxes.append(BoundingBox(x0, y0, x1, y1, float(np.mean(target[component]))))
    return tuple(sorted(boxes, key=lambda item: item.confidence, reverse=True))


def _merge_overlapping_boxes(boxes: Sequence[BoundingBox]) -> Tuple[BoundingBox, ...]:
    """Merge overlapping model regions that belong to one continuous fin/body view."""

    pending = list(boxes)
    merged = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            remainder = []
            for other in pending:
                overlaps = (
                    min(current.x1, other.x1) > max(current.x0, other.x0)
                    and min(current.y1, other.y1) > max(current.y0, other.y0)
                )
                if not overlaps:
                    remainder.append(other)
                    continue
                current = BoundingBox(
                    min(current.x0, other.x0),
                    min(current.y0, other.y0),
                    max(current.x1, other.x1),
                    max(current.y1, other.y1),
                    max(current.confidence, other.confidence),
                )
                changed = True
            pending = remainder
        merged.append(current)
    return tuple(sorted(merged, key=lambda item: item.confidence, reverse=True))


def _fin_boxes_from_edge_probabilities(
    probabilities: np.ndarray,
    original_size: Tuple[int, int],
    coarse_boxes: Sequence[BoundingBox],
    threshold: float = 0.22,
    minimum_size: int = 60,
) -> Tuple[BoundingBox, ...]:
    """Refine coarse dolphin locations to fin-centered boxes from learned edge classes.

    FinFindR's crop network is documented upstream as isolating a dolphin.  Its
    six-class edge network is the available model that localizes the dorsal-fin
    tip, trailing edge and leading edge.  This helper combines those three
    classes and rejects edge components outside the coarse dolphin locations.
    """

    if probabilities.ndim != 3 or probabilities.shape[0] < 6:
        raise FinFindRModelError(
            f"Expected the FinFindR six-class edge output, got {probabilities.shape}."
        )
    if not 0.0 < threshold < 1.0:
        raise ValueError("Fin-edge refinement threshold must be between zero and one.")
    image_width, image_height = original_size
    map_height, map_width = probabilities.shape[1:]
    edge_strength = np.max(probabilities[3:6], axis=0)
    candidate = edge_strength > threshold
    tip_candidate = probabilities[4] > threshold
    leading_candidate = probabilities[5] > threshold

    # The cropper can contain a long, correctly detected dolphin-back contour.
    # Anchor refinement at the learned fin tip, estimate the tip-to-base scale
    # from the closest leading-edge component, then ignore edge pixels farther
    # away than that fin-scale radius. This prevents the back from defining the
    # review crop while retaining the complete fin edge.
    boxes = []
    regions = coarse_boxes or (BoundingBox(0, 0, image_width, image_height, 0.0),)
    for coarse in regions:
        allowed = np.zeros_like(candidate, dtype=bool)
        coarse_x0 = max(0, int(math.floor(coarse.x0 * map_width / image_width)))
        coarse_y0 = max(0, int(math.floor(coarse.y0 * map_height / image_height)))
        coarse_x1 = min(map_width, int(math.ceil(coarse.x1 * map_width / image_width)))
        coarse_y1 = min(map_height, int(math.ceil(coarse.y1 * map_height / image_height)))
        allowed[coarse_y0:coarse_y1, coarse_x0:coarse_x1] = True

        connectivity = np.ones((3, 3), dtype=np.uint8)
        tip_labels, tip_count = ndimage.label(tip_candidate & allowed, structure=connectivity)
        if tip_count == 0:
            continue
        tip_sizes = np.bincount(tip_labels.ravel())
        tip_sizes[0] = 0
        tip_region = tip_labels == int(np.argmax(tip_sizes))
        tip_row, tip_col = _weighted_centroid(probabilities[4], tip_region)

        leading_labels, leading_count = ndimage.label(
            leading_candidate & allowed, structure=connectivity
        )
        if leading_count == 0:
            continue
        closest_leading = None
        closest_distance = float("inf")
        for value in range(1, leading_count + 1):
            rows, cols = np.nonzero(leading_labels == value)
            if len(rows) < 4:
                continue
            distance = float(np.min(np.hypot(rows - tip_row, cols - tip_col)))
            if distance < closest_distance:
                closest_distance = distance
                closest_leading = leading_labels == value
        if closest_leading is None:
            continue
        leading_rows, leading_cols = np.nonzero(closest_leading)
        leading_distances = np.hypot(leading_rows - tip_row, leading_cols - tip_col)
        base_index = int(np.argmax(leading_distances))
        base_row = float(leading_rows[base_index])
        base_col = float(leading_cols[base_index])
        fin_scale = float(leading_distances[base_index])
        if fin_scale < 4.0:
            continue
        grid_rows, grid_cols = np.indices(candidate.shape)
        fin_radius = 1.35 * fin_scale
        radial_distance = np.hypot(grid_rows - tip_row, grid_cols - tip_col)
        near_tip = radial_distance <= fin_radius

        # The trailing class may continue from the fin base along the dolphin
        # back. Separate that continuation at the apex using the learned
        # leading-edge vector: true trailing-edge pixels lie on the opposite
        # angular side from the back extension. Mirroring the image reverses
        # the sign, so infer it from trailing pixels nearest the tip.
        leading_dx = base_col - tip_col
        leading_dy = base_row - tip_row
        cross = leading_dx * (grid_rows - tip_row) - leading_dy * (grid_cols - tip_col)
        trailing_candidate = (probabilities[3] > threshold) & allowed
        sign_probe = trailing_candidate & (radial_distance >= 0.15 * fin_scale) & (
            radial_distance <= 0.65 * fin_scale
        )
        probe_values = cross[sign_probe]
        probe_values = probe_values[np.abs(probe_values) > 1e-6]
        trailing_sign = float(np.sign(np.median(probe_values))) if len(probe_values) else -1.0
        angular_tolerance = 0.04 * fin_scale * fin_scale
        same_angular_side = trailing_sign * cross >= -angular_tolerance
        trailing_support = trailing_candidate & same_angular_side
        support = (
            tip_region
            | closest_leading
            | trailing_support
            | ((probabilities[4] > threshold) & allowed)
        ) & near_tip
        if np.count_nonzero(support) < 12:
            continue
        connected = ndimage.binary_dilation(support, iterations=3)
        rows, cols = np.nonzero(connected)
        raw_width = int(cols.max() - cols.min() + 1)
        raw_height = int(rows.max() - rows.min() + 1)
        if raw_width < 3 or raw_height < 3:
            continue
        margin_x = max(3, int(math.ceil(raw_width * 0.06)))
        margin_y = max(3, int(math.ceil(raw_height * 0.08)))
        x0_map = max(0, int(cols.min()) - margin_x)
        y0_map = max(0, int(rows.min()) - margin_y)
        x1_map = min(map_width, int(cols.max()) + margin_x + 1)
        y1_map = min(map_height, int(rows.max()) + margin_y + 1)
        x0 = int(math.floor(x0_map * image_width / map_width))
        y0 = int(math.floor(y0_map * image_height / map_height))
        x1 = int(math.ceil(x1_map * image_width / map_width))
        y1 = int(math.ceil(y1_map * image_height / map_height))
        if x1 - x0 < minimum_size or y1 - y0 < minimum_size:
            continue
        boxes.append(BoundingBox(
            x0, y0, x1, y1, float(np.mean(edge_strength[support]))
        ))
    return tuple(sorted(boxes, key=lambda item: item.confidence, reverse=True))


def _refine_path_at_image_resolution(
    image: Image.Image,
    guide_points_xy: np.ndarray,
    endpoints_xy: np.ndarray,
    color_boundary_probability: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Follow full-resolution image evidence inside a narrow learned corridor."""

    width, height = image.size
    guide = np.asarray(guide_points_xy, dtype=float)
    endpoints = np.asarray(endpoints_xy, dtype=float)
    if guide.ndim != 2 or guide.shape[1] != 2 or len(guide) < 2:
        raise ValueError("Full-resolution edge refinement requires a two-dimensional guide path.")

    corridor_width = max(12, min(64, int(round(0.015 * max(width, height)))))
    margin = corridor_width + 3
    all_points = np.vstack((guide, endpoints))
    x0 = max(0, int(math.floor(all_points[:, 0].min())) - margin)
    y0 = max(0, int(math.floor(all_points[:, 1].min())) - margin)
    x1 = min(width, int(math.ceil(all_points[:, 0].max())) + margin + 1)
    y1 = min(height, int(math.ceil(all_points[:, 1].max())) + margin + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return endpoints.astype(np.float32)

    local_guide = guide - np.array([x0, y0], dtype=float)
    mask_image = Image.new("L", (x1 - x0, y1 - y0), 0)
    mask_draw = ImageDraw.Draw(mask_image)
    mask_draw.line(
        [tuple(map(float, point)) for point in local_guide],
        fill=255,
        width=2,
        joint="curve",
    )
    guide_mask = np.asarray(mask_image, dtype=np.uint8) > 0
    distance = ndimage.distance_transform_edt(~guide_mask)
    corridor = distance <= corridor_width

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)[y0:y1, x0:x1] / 255.0
    channel_gradients = [
        np.hypot(ndimage.sobel(rgb[:, :, channel], axis=0), ndimage.sobel(rgb[:, :, channel], axis=1))
        for channel in range(3)
    ]
    gradient = np.max(channel_gradients, axis=0)
    gradient /= max(float(np.quantile(gradient[corridor], 0.97)), 1e-6)
    gradient = np.clip(gradient, 0.0, 1.0)

    if color_boundary_probability is not None:
        color_boundary = np.asarray(color_boundary_probability, dtype=np.float32)
        if color_boundary.shape != (height, width):
            raise ValueError("Color-boundary evidence must match the working image size.")
        local_color = color_boundary[y0:y1, x0:x1]
        gradient = np.clip(0.30 * gradient + 0.70 * local_color, 0.0, 1.0)

    cost = 1.0 / np.maximum(gradient, 0.025)
    cost += 0.30 * (distance / max(corridor_width, 1)) ** 2
    cost[~corridor] += 1_000.0
    start = (
        int(round(endpoints[0, 1])) - y0,
        int(round(endpoints[0, 0])) - x0,
    )
    end = (
        int(round(endpoints[1, 1])) - y0,
        int(round(endpoints[1, 0])) - x0,
    )
    start = (min(max(start[0], 0), cost.shape[0] - 1), min(max(start[1], 0), cost.shape[1] - 1))
    end = (min(max(end[0], 0), cost.shape[0] - 1), min(max(end[1], 0), cost.shape[1] - 1))
    cost[start] = min(float(cost[start]), 0.5)
    cost[end] = min(float(cost[end]), 0.5)
    path_rc = _astar(cost, start, end)
    refined = np.column_stack((path_rc[:, 1] + x0, path_rc[:, 0] + y0)).astype(np.float32)
    refined[0] = endpoints[0]
    refined[-1] = endpoints[1]
    return refined


def _edge_path_from_probabilities(
    image: Image.Image,
    probabilities: np.ndarray,
    threshold: float,
    endpoints_xy: Optional[Sequence[Sequence[float]]] = None,
    color_boundary_probability: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    if probabilities.ndim != 3 or probabilities.shape[0] < 6:
        raise FinFindRModelError(
            f"Expected the FinFindR six-class edge output, got {probabilities.shape}."
        )
    original_width, original_height = image.size
    dominant_scale = min(1.0, 1000.0 / max(original_width, original_height))
    pixel_scale = min(1.0, math.sqrt(750_000.0 / (original_width * original_height)))
    trace_scale = min(dominant_scale, pixel_scale)
    trace_size = (
        max(2, round(original_width * trace_scale)),
        max(2, round(original_height * trace_scale)),
    )
    trace_image = image.convert("RGB").resize(trace_size, Image.Resampling.BICUBIC)
    trailing = _resize_probability(probabilities[3], trace_size)
    tip = _resize_probability(probabilities[4], trace_size)
    leading = _resize_probability(probabilities[5], trace_size)
    trail_mask = trailing > threshold
    tip_mask = tip > threshold
    if np.count_nonzero(trail_mask) < 2:
        raise FinFindRModelError("FinFindR did not find a usable trailing-edge region.")

    endpoint_array = None
    if endpoints_xy is not None:
        endpoint_array = np.asarray(endpoints_xy, dtype=float)
        if endpoint_array.shape != (2, 2) or not np.isfinite(endpoint_array).all():
            raise ValueError("Trailing-edge endpoints must be two finite (x, y) points.")
        if (
            np.any(endpoint_array[:, 0] < 0)
            or np.any(endpoint_array[:, 0] >= original_width)
            or np.any(endpoint_array[:, 1] < 0)
            or np.any(endpoint_array[:, 1] >= original_height)
        ):
            raise ValueError("Trailing-edge endpoints must lie inside the working image.")
        start = (
            round(endpoint_array[0, 1] * trace_size[1] / original_height),
            round(endpoint_array[0, 0] * trace_size[0] / original_width),
        )
        end = (
            round(endpoint_array[1, 1] * trace_size[1] / original_height),
            round(endpoint_array[1, 0] * trace_size[0] / original_width),
        )
        start = (min(max(start[0], 0), trace_size[1] - 1), min(max(start[1], 0), trace_size[0] - 1))
        end = (min(max(end[0], 0), trace_size[1] - 1), min(max(end[1], 0), trace_size[0] - 1))
    else:
        focus = _largest_component(ndimage.binary_dilation(trail_mask | tip_mask, iterations=3))
        if not np.any(focus):
            raise FinFindRModelError("FinFindR did not find a connected fin-edge region.")
        tip_region = _largest_component(ndimage.binary_dilation(tip_mask, iterations=2))
        start_float = _weighted_centroid(tip + trailing, tip_region)
        start = tuple(int(round(value)) for value in start_float)

        rows, cols = np.nonzero(trail_mask & ndimage.binary_dilation(focus, iterations=2))
        if rows.size == 0:
            rows, cols = np.nonzero(trail_mask)
        trail_center = np.asarray(_weighted_centroid(trailing, trail_mask))
        lead_center = np.asarray(_weighted_centroid(leading, leading > threshold))
        candidates = np.column_stack((rows, cols)).astype(np.float64)
        scores = (
            np.linalg.norm(candidates - np.asarray(start_float), axis=1)
            + np.linalg.norm(candidates - lead_center, axis=1)
            - 0.25 * np.linalg.norm(candidates - trail_center, axis=1)
        )
        end = tuple(int(value) for value in candidates[int(np.argmax(scores))])

    rgb = np.asarray(trace_image, dtype=np.float32) / 255.0
    channel_gradients = [
        np.hypot(ndimage.sobel(rgb[:, :, channel], axis=0), ndimage.sobel(rgb[:, :, channel], axis=1))
        for channel in range(3)
    ]
    gradient = np.max(channel_gradients, axis=0)
    gradient /= max(float(np.quantile(gradient, 0.97)), 1e-6)
    gradient = np.clip(gradient, 0.0, 1.0)
    if color_boundary_probability is not None:
        color_boundary = np.asarray(color_boundary_probability, dtype=np.float32)
        if color_boundary.shape != (original_height, original_width):
            raise ValueError("Color-boundary evidence must match the working image size.")
        color_trace = np.asarray(
            Image.fromarray(color_boundary, mode="F").resize(trace_size, Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        gradient = np.clip(0.30 * gradient + 0.70 * color_trace, 0.0, 1.0)
    if endpoint_array is None:
        learned = np.clip(0.75 * trailing + 0.20 * tip + 0.05 * gradient, 0.0, 1.0)
        corridor = ndimage.binary_dilation(trail_mask | tip_mask, iterations=8)
    else:
        # The operator's A and C30 landmarks are authoritative endpoints. The
        # model and colour gradient only select the intervening review path.
        learned = np.clip(0.40 * trailing + 0.10 * tip + 0.50 * gradient, 0.0, 1.0)
        endpoint_line = np.zeros_like(trailing, dtype=bool)
        line_rows = np.rint(np.linspace(start[0], end[0], 400)).astype(int)
        line_cols = np.rint(np.linspace(start[1], end[1], 400)).astype(int)
        endpoint_line[line_rows, line_cols] = True
        margin = max(12, round(0.18 * max(trace_size)))
        corridor = (
            ndimage.binary_dilation(trail_mask | tip_mask, iterations=12)
            | ndimage.binary_dilation(endpoint_line, iterations=margin)
        )
    cost = 1.0 / np.maximum(learned, 0.02)
    cost[~corridor] += 35.0
    cost[start] = min(float(cost[start]), 0.5)
    cost[end] = min(float(cost[end]), 0.5)
    active_rows, active_cols = np.nonzero(corridor)
    margin = 12
    row0 = max(0, min(int(active_rows.min()), start[0], end[0]) - margin)
    row1 = min(cost.shape[0], max(int(active_rows.max()), start[0], end[0]) + margin + 1)
    col0 = max(0, min(int(active_cols.min()), start[1], end[1]) - margin)
    col1 = min(cost.shape[1], max(int(active_cols.max()), start[1], end[1]) + margin + 1)
    local_start = (start[0] - row0, start[1] - col0)
    local_end = (end[0] - row0, end[1] - col0)
    path_rc = _astar(cost[row0:row1, col0:col1], local_start, local_end)
    path_rc[:, 0] += row0
    path_rc[:, 1] += col0

    scale_x = original_width / trace_size[0]
    scale_y = original_height / trace_size[1]
    points_xy = np.column_stack((path_rc[:, 1] * scale_x, path_rc[:, 0] * scale_y))
    if endpoint_array is not None:
        points_xy[0] = endpoint_array[0]
        points_xy[-1] = endpoint_array[1]
        points_xy = _refine_path_at_image_resolution(
            image,
            points_xy,
            endpoint_array,
            color_boundary_probability=color_boundary_probability,
        )
    confidence = float(np.mean(trailing[path_rc[:, 0], path_rc[:, 1]]))
    return points_xy.astype(np.float32), confidence


class FinFindRAssistant:
    """Lazily loaded, inference-only access to FinFindR assistance."""

    def __init__(self, model_directory: Optional[Path] = None, device: str = "cpu"):
        self.bundle: ModelBundle = locate_model_bundle(model_directory)
        self.device = device
        self._cropper = None
        self._edge_model = None
        self._hashes = model_hashes(self.bundle.directory)

    @property
    def provenance(self) -> Dict[str, object]:
        return {
            "upstream": "FinFindR 0.1.21",
            "upstream_repository": "https://github.com/haimeh/finFindR",
            "upstream_commit": self.bundle.upstream_commit,
            "model_sha256": dict(self._hashes),
            "inference": "FinShape Surfer inference-only MXNet symbol adapter",
            "scientific_status": "operator-reviewed assistance; not validated automatic measurement",
        }

    def _crop_model(self) -> MXNetSymbolExecutor:
        if self._cropper is None:
            self._cropper = MXNetSymbolExecutor(
                self.bundle.cropper_symbol, self.bundle.cropper_params, self.device
            )
        return self._cropper

    def _trailing_model(self) -> MXNetSymbolExecutor:
        if self._edge_model is None:
            self._edge_model = MXNetSymbolExecutor(
                self.bundle.edge_symbol, self.bundle.edge_params, self.device
            )
        return self._edge_model

    def detect_fins(
        self,
        image: Image.Image,
        threshold: float = 0.45,
        minimum_size: int = 100,
        include_body: bool = False,
    ) -> FinDetectionResult:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Detection threshold must be between zero and one.")
        original = image.convert("RGB")
        resized = original.resize((225, 150), Image.Resampling.BICUBIC)
        network_input = _nchw(resized)
        maximum = float(network_input.max())
        if maximum > 0:
            network_input /= maximum
        model = self._crop_model()
        raw = model.predict(network_input)
        probabilities = raw[0]
        # FinFindR's R API is one-based: cropFins(target=1) selects Python
        # channel zero. Channel one is not the fin class.
        target_channels = (0, 1) if include_body else (0,)
        boxes = _merge_overlapping_boxes(_crop_boxes_from_probabilities(
            probabilities, original.size, threshold, target_channels, minimum_size
        ))
        tile_count = 0
        if not boxes:
            tiled_boxes = []
            for x0, y0, x1, y1 in _fallback_detection_tiles(original.size):
                tile_count += 1
                tile = original.crop((x0, y0, x1, y1))
                tile_input = _nchw(tile.resize((225, 150), Image.Resampling.BICUBIC))
                tile_maximum = float(tile_input.max())
                if tile_maximum > 0:
                    tile_input /= tile_maximum
                tile_probabilities = model.predict(tile_input)[0]
                local = _crop_boxes_from_probabilities(
                    tile_probabilities,
                    tile.size,
                    threshold,
                    target_channels,
                    minimum_size,
                )
                tiled_boxes.extend(_offset_box(box, x0, y0) for box in local)
            boxes = _merge_overlapping_boxes(tiled_boxes)
        provenance = dict(self.provenance)
        provenance.update({
            "operation": "automatic fin crop suggestion",
            "postprocessing": (
                "FinFindR blobber.R-compatible morphology and bounding boxes; "
                "overlapping regions merged for one review crop"
            ),
            "r_target": 1,
            "python_target_channel": 0,
            "target": "body_and_fin" if include_body else "fin",
            "multiscale_fallback_tiles": tile_count,
        })
        return FinDetectionResult(boxes, probabilities[target_channels[-1]], threshold, provenance)

    def detect_fin_crop(
        self,
        image: Image.Image,
        coarse_threshold: float = 0.45,
        edge_threshold: float = 0.18,
        minimum_size: int = 60,
    ) -> FinDetectionResult:
        """Return a fin-centered review crop using both FinFindR model stages."""

        original = image.convert("RGB")
        coarse = self.detect_fins(
            original,
            threshold=coarse_threshold,
            minimum_size=minimum_size,
            include_body=False,
        )
        edge_model = self._trailing_model()
        refined_boxes = []
        probability_maps = []
        regions = [
            (box.x0, box.y0, box.x1, box.y1)
            for box in coarse.boxes
        ]
        if not regions:
            regions = list(_fallback_detection_tiles(original.size))
        for x0, y0, x1, y1 in regions:
            margin_x = int(round((x1 - x0) * 0.08))
            margin_y = int(round((y1 - y0) * 0.08))
            x0 = max(0, x0 - margin_x)
            y0 = max(0, y0 - margin_y)
            x1 = min(original.width, x1 + margin_x)
            y1 = min(original.height, y1 + margin_y)
            region = original.crop((x0, y0, x1, y1))
            resized, _scale_x, _scale_y = _resize_dominant(region, 240)
            edge_probabilities = edge_model.predict(_nchw(resized))[0]
            probability_maps.append(np.max(edge_probabilities[3:6], axis=0))
            local_boxes = _fin_boxes_from_edge_probabilities(
                edge_probabilities,
                region.size,
                (),
                threshold=edge_threshold,
                minimum_size=minimum_size,
            )
            refined_boxes.extend(_offset_box(box, x0, y0) for box in local_boxes)
        boxes = _merge_overlapping_boxes(refined_boxes)
        probability_map = (
            max(probability_maps, key=lambda values: float(np.max(values)))
            if probability_maps else np.zeros((1, 1), dtype=np.float32)
        )
        provenance = dict(self.provenance)
        provenance.update({
            "operation": "automatic fin-centered crop suggestion",
            "coarse_stage": "FinFindR cropFins dolphin isolator, R target 1 / Python channel 0",
            "refinement_stage": "FinFindR trailing, tip and leading-edge classes 4-6",
            "coarse_box_count": len(coarse.boxes),
            "edge_inference_region_count": len(regions),
            "edge_inference": "crop-local 240-pixel inference with tiled fallback",
            "edge_threshold": edge_threshold,
            "fallback_to_dolphin_box": False,
            "warning": "Fin-only crop is an operator-reviewed suggestion, not a measurement.",
        })
        return FinDetectionResult(boxes, probability_map, edge_threshold, provenance)

    def suggest_trailing_edge(
        self,
        image: Image.Image,
        threshold: float = 0.35,
        endpoints_xy: Optional[Sequence[Sequence[float]]] = None,
    ) -> TrailingEdgeResult:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Edge threshold must be between zero and one.")
        original = image.convert("RGB")
        resized, _scale_x, _scale_y = _resize_dominant(original, 200)
        raw = self._trailing_model().predict(_nchw(resized))
        probabilities = raw[0]
        points, confidence = _edge_path_from_probabilities(
            original, probabilities, threshold, endpoints_xy=endpoints_xy
        )
        provenance = dict(self.provenance)
        provenance.update({
            "operation": "machine-learned trailing-edge suggestion",
            "postprocessing": "FinShape Surfer weighted A* adapter",
            "learned_channels": {"trailing": 4, "tip": 5, "leading": 6},
            "operator_endpoint_constraint": (
                "A-to-C30" if endpoints_xy is not None else "not supplied"
            ),
            "pixel_resolution_refinement": endpoints_xy is not None,
            "pixel_resolution_refinement_method": (
                "full-resolution RGB gradient A* inside the learned edge corridor"
                if endpoints_xy is not None else "not applied"
            ),
            "warning": "Review and correct this suggestion before landmarking or reconstruction.",
        })
        return TrailingEdgeResult(
            points=points,
            confidence=confidence,
            trailing_probability=probabilities[3],
            tip_probability=probabilities[4],
            provenance=provenance,
        )

    def suggest_fin_contour(
        self,
        image: Image.Image,
        *,
        b_xy: Sequence[float],
        a_xy: Sequence[float],
        c30_xy: Sequence[float],
        tissue_polygon_xy: Optional[Sequence[Sequence[float]]] = None,
        threshold: float = 0.35,
    ) -> FinContourResult:
        """Trace dense B-to-A-to-C30 Photo-ID edges from one model pass.

        FinFindR's learned edge classes guide the paths while A, B and C30
        remain exact operator landmarks. This result never replaces measured
        Bezier or CFD geometry.
        """

        if not 0.0 < threshold < 1.0:
            raise ValueError("Edge threshold must be between zero and one.")
        anchors = np.asarray((b_xy, a_xy, c30_xy), dtype=float)
        if anchors.shape != (3, 2) or not np.isfinite(anchors).all():
            raise ValueError("A, B and C30 must be finite two-dimensional points.")

        original = image.convert("RGB")
        resized, _scale_x, _scale_y = _resize_dominant(original, 200)
        probabilities = self._trailing_model().predict(_nchw(resized))[0]
        color_boundary = _robust_color_boundary_probability(original, tissue_polygon_xy)

        leading_probabilities = np.array(probabilities, copy=True)
        leading_probabilities[3] = probabilities[5]
        leading, leading_confidence = _edge_path_from_probabilities(
            original,
            leading_probabilities,
            threshold,
            endpoints_xy=(anchors[0], anchors[1]),
            color_boundary_probability=color_boundary,
        )
        trailing, trailing_confidence = _edge_path_from_probabilities(
            original,
            probabilities,
            threshold,
            endpoints_xy=(anchors[1], anchors[2]),
            color_boundary_probability=color_boundary,
        )
        leading = _smooth_leading_path(np.asarray(leading, dtype=np.float64))
        trailing = np.asarray(trailing, dtype=np.float64)
        leading[0], leading[-1] = anchors[0], anchors[1]
        trailing[0], trailing[-1] = anchors[1], anchors[2]
        outline = np.vstack((leading, trailing[1:]))
        outline = outline[np.r_[True, np.any(np.diff(outline, axis=0) != 0, axis=1)]]
        outline[0] = anchors[0]
        outline[np.argmin(np.linalg.norm(outline - anchors[1], axis=1))] = anchors[1]
        outline[-1] = anchors[2]

        provenance = dict(self.provenance)
        provenance.update({
            "operation": "machine-learned full fin contour suggestion for Photo-ID masking",
            "learned_channels": {"trailing": 4, "tip": 5, "leading": 6},
            "operator_endpoint_constraints": "B-to-A leading and A-to-C30 trailing",
            "postprocessing": (
                "two full-resolution color-and-gradient A* paths inside learned corridors; "
                "leading path receives subpixel anti-staircase smoothing"
            ),
            "color_boundary_model": (
                "robust tissue/water seeds from eroded operator Bezier interior and outer frame"
                if color_boundary is not None else "unavailable; RGB gradients only"
            ),
            "photo_id_mask_only": True,
            "measurement_geometry_changed": False,
            "cfd_surface_changed": False,
        })
        return FinContourResult(
            leading_points=leading,
            trailing_points=trailing,
            outline_points=outline,
            leading_confidence=leading_confidence,
            trailing_confidence=trailing_confidence,
            provenance=provenance,
        )


def draw_assistance_preview(
    image: Image.Image,
    detections: Optional[FinDetectionResult] = None,
    edge: Optional[TrailingEdgeResult] = None,
) -> Image.Image:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    if detections is not None:
        for box in detections.boxes:
            draw.rectangle((box.x0, box.y0, box.x1, box.y1), outline=(0, 220, 90), width=3)
    if edge is not None and len(edge.points) > 1:
        draw.line([tuple(map(float, point)) for point in edge.points], fill=(255, 140, 0), width=3)
    return preview
