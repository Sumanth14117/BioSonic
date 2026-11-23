"""Small atomic-write and integrity helpers for M5-only artifacts."""

import csv
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(data: bytes, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Preserving existing temporary output: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    atomic_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"), path)


def atomic_text(text: str, path: Path) -> None:
    atomic_bytes(text.encode("utf-8"), path)


def atomic_csv(rows, fieldnames, path: Path) -> None:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(buffer.getvalue(), path)


def atomic_npz(path: Path, **arrays) -> None:
    import io
    import numpy as np

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(buffer.getvalue(), path)


def atomic_joblib_dump(obj, path: Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Preserving existing temporary model output: {temporary}")
    joblib.dump(obj, temporary, compress=3)
    os.replace(temporary, path)
