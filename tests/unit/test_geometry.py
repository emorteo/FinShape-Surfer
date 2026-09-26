import math
import numpy as np
from core.math.geometry import (
    clamp, v_add, v_sub, v_mul, v_len, v_norm, dist, shoelace_area, rot2, rotate_about_origin,
    rot_np, normalize_np, y_on_line_through_np, polygon_area_np, x_intersections_with_horizontal_np
)

def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10

def test_tuple_vector_math():
    a = (3.0, 4.0)
    b = (1.0, 2.0)
    assert v_add(a, b) == (4.0, 6.0)
    assert v_sub(a, b) == (2.0, 2.0)
    assert v_mul(a, 2.0) == (6.0, 8.0)
    assert v_len(a) == 5.0
    assert v_norm(a) == (3.0/5.0, 4.0/5.0)
    assert dist(a, b) == math.hypot(2.0, 2.0)

def test_shoelace_area():
    # Square 0,0 to 2,2
    poly = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    assert math.isclose(shoelace_area(poly), 4.0)
    assert shoelace_area([(0.0, 0.0)]) == 0.0

def test_rot2():
    v = (1.0, 0.0)
    rot_v = rot2(v, math.pi / 2)
    assert math.isclose(rot_v[0], 0.0, abs_tol=1e-9)
    assert math.isclose(rot_v[1], 1.0, abs_tol=1e-9)

def test_numpy_math():
    v = np.array([1.0, 0.0])
    r = rot_np(v, math.pi / 2)
    assert np.allclose(r, [0.0, 1.0])
    
    n = normalize_np(np.array([3.0, 4.0]))
    assert np.allclose(n, [0.6, 0.8])
    
    B = np.array([0.0, 0.0])
    A = np.array([2.0, 4.0])
    y = y_on_line_through_np(B, A, 1.0)
    assert math.isclose(y, 2.0)

def test_polygon_area_np():
    poly = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    assert math.isclose(polygon_area_np(poly), 4.0)

def test_x_intersections_with_horizontal_np():
    poly = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 0.0]])
    xs = x_intersections_with_horizontal_np(poly, 1.0)
    assert len(xs) == 2
    assert math.isclose(xs[0], 1.0)
    assert math.isclose(xs[1], 3.0)
