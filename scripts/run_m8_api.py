"""Start the local M8 API after its documented API-only dependencies are available."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    missing = [name for name in ("fastapi", "uvicorn", "multipart") if importlib.util.find_spec(name) is None]
    if missing:
        print("M8 API dependencies missing: " + ", ".join(missing) + ". See requirements-api.txt; no automatic installation is attempted.", file=sys.stderr)
        return 2
    import uvicorn

    uvicorn.run("biosonic_api.main:app", host="127.0.0.1", port=8000, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
