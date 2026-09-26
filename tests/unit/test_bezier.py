import numpy as np
from core.math.bezier import (
    bezier_quad, bezier_cubic, sample_curve, bezier_quad_np, bezier_cubic_np
)

def test_bezier_quad_tuple():
    q0 = (0.0, 0.0)
    q1 = (1.0, 1.0)
    q2 = (2.0, 0.0)
    p = bezier_quad(q0, q1, q2, 0.5)
    assert p == (1.0, 0.5)

def test_bezier_cubic_tuple():
    p0 = (0.0, 0.0)
    p1 = (1.0, 2.0)
    p2 = (2.0, -2.0)
    p3 = (3.0, 0.0)
    p = bezier_cubic(p0, p1, p2, p3, 0.5)
    assert p == (1.5, 0.0)

def test_sample_curve():
    q0 = (0.0, 0.0)
    q1 = (1.0, 1.0)
    q2 = (2.0, 0.0)
    pts = sample_curve(lambda t: bezier_quad(q0, q1, q2, t), n=3)
    assert len(pts) == 3
    assert pts[0] == (0.0, 0.0)
    assert pts[1] == (1.0, 0.5)
    assert pts[2] == (2.0, 0.0)

def test_bezier_quad_np():
    q0 = np.array([0.0, 0.0])
    q1 = np.array([1.0, 1.0])
    q2 = np.array([2.0, 0.0])
    t = np.array([0.0, 0.5, 1.0])
    pts = bezier_quad_np(q0, q1, q2, t)
    assert np.allclose(pts[0], [0.0, 0.0])
    assert np.allclose(pts[1], [1.0, 0.5])
    assert np.allclose(pts[2], [2.0, 0.0])

def test_bezier_cubic_np():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 2.0])
    p2 = np.array([2.0, -2.0])
    p3 = np.array([3.0, 0.0])
    t = np.array([0.0, 0.5, 1.0])
    pts = bezier_cubic_np(p0, p1, p2, p3, t)
    assert np.allclose(pts[0], [0.0, 0.0])
    assert np.allclose(pts[1], [1.5, 0.0])
    assert np.allclose(pts[2], [3.0, 0.0])
