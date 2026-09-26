import pandas as pd
from pathlib import Path
from core.io.data_loader import get_any, save_measurements_csv

def test_get_any():
    s = pd.Series({"a": 1.0, "b": 2.0})
    assert get_any(s, ["c", "a"]) == 1.0
    assert get_any(s, ["c", "d"]) is None

def test_save_measurements_csv(tmp_path):
    p = tmp_path / "test.csv"
    data = [{"col1": "a", "col2": "b"}]
    save_measurements_csv(p, ["col1", "col2"], data)
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) == 1
    assert df["col1"].iloc[0] == "a"
