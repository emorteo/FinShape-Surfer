import pandas as pd
from pathlib import Path
from finshape3d import run

def test_finshape3d_run(tmp_path):
    # Create dummy csv
    csv_file = tmp_path / "dummy.csv"
    df = pd.DataFrame({
        "id": ["test_fin"],
        "r_BA": [1.0],
        "r_BC5": [0.85],
        "r_BC10": [0.4],
        "r_BC20": [0.4],
        "r_BC30": [1.0],
    })
    df.to_csv(csv_file, index=False)

    outdir = tmp_path / "out"
    
    # Run pipeline
    run(
        input_path=csv_file,
        outdir=outdir,
        bc30scale=10.0,
        y_slices=5,
        z_slices=3,
        top_focus_power=1.0,
        montage=False
    )

    # Check if files were created
    assert (outdir / "FinShape3D_1.1.9_2D_test_fin.png").exists()
    assert (outdir / "FinShape3D_1.1.9_3D_test_fin.png").exists()
