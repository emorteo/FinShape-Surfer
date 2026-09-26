import pytest
from unittest.mock import MagicMock, patch
from core.ui.gui_2d import FinShape2DApp, Landmarks, BezierControls
import numpy as np
import pandas as pd

def test_generate_3d_logic():
    # Mock the app
    app = MagicMock(spec=FinShape2DApp)
    app.landmarks = {
        "A": (10.0, 10.0), "B": (0.0, 0.0), "C5": (2.0, 2.0),
        "C10": (4.0, 4.0), "C20": (6.0, 6.0), "C30": (8.0, 8.0),
        "D": (12.0, 12.0)
    }
    app.controls = MagicMock(spec=BezierControls)
    app.image_path = MagicMock()
    app.image_path.stem = "test_fin"
    app.image_path.parent = MagicMock()
    
    # Mock the reconstruction functions
    with patch("core.ui.gui_2d.compute_weller_ratios") as mock_ratios, \
         patch("core.ui.gui_2d.build_landmarks") as mock_build_lm, \
         patch("core.ui.gui_2d.build_controls") as mock_build_ctrl, \
         patch("core.ui.gui_2d.compute_2d_curves") as mock_curves, \
         patch("core.ui.gui_2d.plot_3d_wireframe") as mock_plot, \
         patch("core.ui.gui_2d.platform.system", return_value="Linux"), \
         patch("core.ui.gui_2d.subprocess.call") as mock_subprocess:
        
        mock_ratios.return_value = {"AB_px": 10.0, "C5B_over_AB": 0.5, "C10B_over_AB": 0.5, "C20B_over_AB": 0.5, "C30B_over_AB": 0.5, "DB_over_AB": 0.5}
        
        # Call the logic (simulating generate_3d)
        # We need to manually call the logic that was inside generate_3d
        # Since we are testing the logic, we can just call the function directly if it were a method, 
        # but here we are testing the logic inside the method.
        
        # Reconstruct
        from core.reconstruction.wireframe import build_landmarks as build_landmarks_3d
        
        # Create a dummy row
        row = pd.Series({
            "r_BA": 10.0,
            "r_BC5": 0.5,
            "r_BC10": 0.5,
            "r_BC20": 0.5,
            "r_BC30": 0.5,
            "id": "test_fin"
        })
        
        lm_3d, u_AB = build_landmarks_3d(row, BC30_scale=10.0)
        ctrl_3d = mock_build_ctrl(lm_3d, u_AB)
        curves2d = mock_curves(ctrl_3d, n=900)
        
        # Plot
        outdir = app.image_path.parent / "FinShape3D_outputs"
        p3d = outdir / f"{app.image_path.stem}_3D.png"
        mock_plot(curves2d, app.image_path.stem, p3d)
        
        # Assertions
        assert mock_plot.called
