"""I/O helpers for reading/writing measurement tables.

This module centralizes CSV/Excel reading and provides a small utility
`get_any` that looks up multiple possible column names from a pandas
Series and returns the first present numeric value. Use `save_measurements_csv`
to persist measurement rows with a stable header ordering.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import csv

def read_table(path: Path) -> pd.DataFrame:
    """Read a table from CSV or Excel into a DataFrame.

    The function infers format from the file suffix.
    """
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def get_any(row: pd.Series, keys: List[str]) -> Optional[float]:
    """Return the first available numeric value from `row` among `keys`.

    This helper simplifies callers that accept multiple alternate column
    names (e.g. backward-compatible header names).
    """
    for k in keys:
        if k in row.index and pd.notna(row[k]):
            return float(row[k])
    return None

def save_measurements_csv(path: Path, columns: List[str], measurements: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in measurements:
            w.writerow(row)
