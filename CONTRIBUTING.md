## Contributing to FinShape3D

Thanks for your interest in contributing. Quick notes to get started:

- Development environment:
  - Python 3.10+ recommended (3.8+ supported).
  - Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- Optional system dependencies:
  - `tkinter` (system Python package) for the GUI.
  - `ghostscript` may be required for some PostScript raster workflows.
  - On Linux, install `xdg-utils` to enable automatic file opening (`xdg-open`).

- Running tests:
  - `pytest` (included in `requirements.txt`) — run `pytest -q`.

- CI: a GitHub Actions workflow runs tests on Linux/macOS/Windows.

When opening a PR, please include a short description, link tests, and keep changes focused.
