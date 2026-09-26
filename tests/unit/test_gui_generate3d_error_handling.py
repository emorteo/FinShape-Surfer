import types
import pandas as pd
from unittest.mock import patch

from core.ui.gui_2d import FinShape2DApp


def make_dummy_self():
    # minimal object with attributes used by generate_3d
    dummy = types.SimpleNamespace()
    dummy.locked_after_save = True
    dummy.controls = object()
    dummy.landmarks = {
        "B": (0.0, 0.0), "A": (1.0, 0.0), "C30": (-10.0, 0.0),
        "C20": (-6.0, 0.0), "C10": (-4.0, 0.0), "C5": (-2.0, 0.0), "D": (2.0, 1.0)
    }

    img = types.SimpleNamespace()
    img.stem = "test_fin"
    from pathlib import Path
    img.parent = Path(".")
    dummy.image_path = img

    # status used later but not required for this test path
    dummy.status = types.SimpleNamespace()
    dummy.status.set = lambda s: None

    return dummy


def test_generate3d_handles_build_controls_error():
    dummy = make_dummy_self()

    # prepare a fake ratios dict returned by compute_weller_ratios
    fake_ratios = {
        "C30B_over_AB": 0.5,
        "C5B_over_AB": 0.5,
        "C10B_over_AB": 0.4,
        "C20B_over_AB": 0.3,
    }

    with patch("core.ui.gui_2d.compute_weller_ratios", return_value=fake_ratios), \
         patch("core.reconstruction.wireframe.build_landmarks", return_value=({"B": None}, None)), \
         patch("core.ui.gui_2d.build_controls", side_effect=ValueError("TE3_P1 must lie within x-span of segment C20–C30.")) as mock_build, \
         patch("tkinter.messagebox.showerror") as mock_msg:

        # call the unbound method with our dummy self
        FinShape2DApp.generate_3d(dummy)

        # Ensure our mocked build_controls was invoked and the error was reported
        assert mock_build.called
        assert mock_msg.called
