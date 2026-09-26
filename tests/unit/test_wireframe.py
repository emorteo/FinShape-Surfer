import math
import numpy as np
import pandas as pd
from core.reconstruction.wireframe import (
    morteo_SP_T, thickness_piecewise, invert_thickness_for_c, build_landmarks
)

def test_morteo_sp_t():
    SP, T = morteo_SP_T(10.0)
    assert math.isclose(SP, 0.4357 * 10.0 - 1.8803)
    assert math.isclose(T, 0.1191 * 10.0 + 0.5164)

def test_thickness_piecewise():
    c = np.array([2.0, 5.0, 8.0])
    C = 10.0
    SP = 4.0
    T = 2.0
    th = thickness_piecewise(c, C, SP, T)
    assert th.shape == (3,)
    assert th[1] > 0

def test_invert_thickness():
    target = 1.0
    C = 10.0
    SP = 4.0
    T = 2.0
    inv = invert_thickness_for_c(target, C, SP, T)
    assert inv is not None
    cL, cR = inv
    assert 0 <= cL <= SP
    assert SP <= cR <= C

def test_build_landmarks():
    row = pd.Series({
        "r_BA": 1.0,
        "r_BC5": 0.8,
        "r_BC10": 0.6,
        "r_BC20": 0.4,
        "r_BC30": 1.0
    })
    lm, u_AB = build_landmarks(row, BC30_scale=10.0)
    assert "B" in lm
    assert "C30" in lm
    assert lm["C30"][0] == -10.0
