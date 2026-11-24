"""Acquire only the annotated BirdCLEF+ 2026 M7 soundscape recordings."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from biosonic_ml.m7_soundscape import load_soundscape_annotations
from download_approved_audio import (
    MAX_FILE_ATTEMPTS,
    SUCCESS_REQUEST_DELAY_SECONDS,
    SafeKaggleApi,
    UnsafePartialResumeError,
    inspect_audio,
    inspect_staged_partial,
    safe_error,
)


COMPETITION = "birdclef-2026"
ANNOTATION_PATH = ROOT / "data/metadata/train_soundscapes_labels.csv"
TAXONOMY_PATH = ROOT / "data/metadata/taxonomy.csv"
INVENTORY_PATH = ROOT / "data/metadata/birdclef_file_inventory.csv"
TRAIN_METADATA_PATH = ROOT / "data/metadata/train.csv"
DATA_ROOT = ROOT / "data"
FINAL_ROOT = DATA_ROOT / "raw/train_soundscapes"
STAGING_ROOT = DATA_ROOT / "raw/.birdclef_m7_soundscapes_staging"
MANIFEST_PATH = DATA_ROOT / "processed/m7_soundscape/soundscape_manifest.csv"
REPORT_PATH = ROOT / "reports/ml/m7_soundscape/soundscape_acquisition_report.md"
EXPECTED_FILES = 66
EXPECTED_BYTES = 33_785_154
MANIFEST_COLUMNS = [
    "filename", "source_path", "expected_size_bytes", "actual_size_bytes",
    "download_status", "verification_status", "sample_rate", "channels",
    "duration_seconds", "audio_format", "sha256", "error_message",
]


def _safe_filename(filename: str) -> bool:
    return (
        bool(filename)
        and re.fullmatch(r"[A-Za-z0-9_.-]+", filename) is not None
        and Path(filename).name == filename
        and "/" not in filename
        and "\\" not in filename
        and ".." not in Path(filename).parts
        and filename.lower().endswith(".ogg")
    )


def build_manifest() -> pd.DataFrame:
    """Derive the exact annotation filename set and inventory sizes."""
    taxonomy = pd.read_csv(TAXONOMY_PATH, dtype=str)
    annotations = load_soundscape_annotations(ANNOTATION_PATH, taxonomy)
    filenames = sorted(annotations.filename.unique().tolist())
    if len(filenames) != EXPECTED_FILES or any(not _safe_filename(name) for name in filenames):
        raise RuntimeError(f"Expected {EXPECTED_FILES} safe unique annotation filenames; found {len(filenames)}")

    inventory = pd.read_csv(INVENTORY_PATH, dtype={"FileName": str})
    soundscape_rows = inventory[inventory.FileName.astype(str).str.startswith("train_soundscapes/")]
    size_map: dict[str, int] = {}
    for row in soundscape_rows.itertuples(index=False):
        relative = str(row.FileName).replace("\\", "/")
        if relative in size_map:
            raise RuntimeError(f"Duplicate inventory path for {relative}")
        size_map[relative] = int(row.SizeBytes)

    rows = []
    for filename in filenames:
        source_path = f"train_soundscapes/{filename}"
        if source_path not in size_map:
            raise RuntimeError(f"Annotated filename has no exact inventory entry: {source_path}")
        rows.append({
            "filename": filename,
            "source_path": source_path,
            "expected_size_bytes": size_map[source_path],
            "actual_size_bytes": pd.NA,
            "download_status": "not_attempted",
            "verification_status": "not_attempted",
            "sample_rate": pd.NA,
            "channels": pd.NA,
            "duration_seconds": pd.NA,
            "audio_format": "",
            "sha256": "",
            "error_message": "",
        })
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if len(frame) != EXPECTED_FILES or frame.filename.nunique() != EXPECTED_FILES:
        raise RuntimeError("The approved M7 manifest is not exactly 66 unique files")
    if int(frame.expected_size_bytes.sum()) != EXPECTED_BYTES:
        raise RuntimeError("M7 expected byte total differs from the validated inventory")

    if MANIFEST_PATH.exists():
        existing = pd.read_csv(MANIFEST_PATH, dtype={"filename": str, "source_path": str})
        if set(existing.columns) != set(MANIFEST_COLUMNS):
            raise RuntimeError("Existing M7 manifest schema differs; refusing to overwrite")
        if existing.filename.nunique() != EXPECTED_FILES or set(existing.filename) != set(frame.filename):
            raise RuntimeError("Existing M7 manifest whitelist differs; refusing to overwrite")
        existing_sizes = existing.set_index("filename").expected_size_bytes.astype("int64").to_dict()
        expected_sizes = frame.set_index("filename").expected_size_bytes.astype("int64").to_dict()
        existing_sources = existing.set_index("filename").source_path.to_dict()
        expected_sources = frame.set_index("filename").source_path.to_dict()
        if existing_sizes != expected_sizes or existing_sources != expected_sources:
            raise RuntimeError("Existing M7 manifest inventory mapping differs; refusing to overwrite")
        prior = existing.set_index("filename")
        frame = frame.set_index("filename")
        for column in MANIFEST_COLUMNS:
            if column != "filename":
                frame[column] = prior[column].reindex(frame.index).combine_first(frame[column])
        frame = frame.reset_index()[MANIFEST_COLUMNS]
    return frame


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False,
                                     dir=path.parent, prefix=path.name + ".", suffix=".tmp") as stream:
        name = stream.name
        frame.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audio_fields(path: Path, expected_size: int) -> tuple[dict[str, object] | None, str]:
    actual = path.stat().st_size
    if actual != expected_size:
        return None, f"File size {actual} does not match inventory size {expected_size}"
    valid, error, metadata = inspect_audio(path)
    if not valid:
        return None, error or "SoundFile could not decode the OGG"
    metadata["actual_size_bytes"] = actual
    metadata["sha256"] = _sha256(path)
    return metadata, ""


def _mark_verified(frame: pd.DataFrame, index: int, status: str, fields: dict[str, object]) -> None:
    frame.at[index, "download_status"] = status
    frame.at[index, "verification_status"] = "verified"
    frame.at[index, "error_message"] = ""
    for column, value in fields.items():
        frame.at[index, column] = value


def _is_stop_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) in (401, 403, 429):
            return True
        current = current.__cause__ or current.__context__
    message = safe_error(exc).lower()
    return isinstance(exc, (ConnectionError, TimeoutError, OSError)) or any(
        token in message for token in (
            "retrybudgetexceeded", "name resolution", "failed to resolve", "connectionerror",
            "connecttimeout", "readtimeout", "too many requests", "http 401", "http 403",
            "http 429", "authentication", "permission/access", "max retries exceeded",
        )
    )


def _write_report(frame: pd.DataFrame, stop_reason: str, unexpected_final: list[str], staged_count: int) -> None:
    verified = frame[frame.verification_status.eq("verified")]
    expected_total = int(frame.expected_size_bytes.sum())
    actual_total = int(verified.actual_size_bytes.fillna(0).sum())
    downloaded = int(frame.download_status.eq("downloaded").sum())
    failed = int(frame.verification_status.isin(["failed", "deferred"]).sum())
    pending = int(frame.verification_status.eq("not_attempted").sum())
    rates = sorted(int(value) for value in verified.sample_rate.dropna().unique())
    channels = sorted(int(value) for value in verified.channels.dropna().unique())
    formats = sorted(str(value) for value in verified.audio_format.dropna().unique())
    duration = (
        f"{verified.duration_seconds.min():.3f}–{verified.duration_seconds.max():.3f} s"
        if not verified.empty else "unavailable"
    )
    lines = [
        "# M7 Annotated Soundscape Acquisition Report", "",
        f"- Approved files: {len(frame)}", f"- Downloaded by this workflow: {downloaded}",
        f"- Successfully verified and present: {len(verified)}", f"- Failed or deferred: {failed}",
        f"- Pending/not attempted: {pending}", f"- Expected total: {expected_total:,} bytes",
        f"- Actual bytes for verified files: {actual_total:,} bytes",
        f"- Newly created final files outside the approved set: {len(unexpected_final)}",
        f"- Staging files remaining (including resumable partials): {staged_count}",
        f"- Verified sample rates: {rates or 'unavailable'} Hz",
        f"- Verified channel counts: {channels or 'unavailable'}",
        f"- Verified format/subtype: {formats or 'unavailable'}",
        f"- Verified duration range: {duration}",
        f"- SHA-256 manifest: [`soundscape_manifest.csv`](../../../data/processed/m7_soundscape/soundscape_manifest.csv)",
        "- Source: individually requested files from Kaggle competition `birdclef-2026`.",
        "- No complete competition archive or unannotated soundscape was requested.",
        "", "## Per-file verification", "",
        "| Filename | Expected bytes | Actual bytes | Download status | Verification | Sample rate | Channels | Duration (s) |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        actual = "" if pd.isna(row.actual_size_bytes) else str(int(row.actual_size_bytes))
        rate = "" if pd.isna(row.sample_rate) else str(int(row.sample_rate))
        channel = "" if pd.isna(row.channels) else str(int(row.channels))
        dur = "" if pd.isna(row.duration_seconds) else f"{float(row.duration_seconds):.3f}"
        lines.append(
            f"| `{row.filename}` | {int(row.expected_size_bytes):,} | {actual} | "
            f"{row.download_status} | {row.verification_status} | {rate} | {channel} | {dur} |"
        )
    lines.extend([
        "", "Each file is accepted only after inventory byte-size equality and a complete SoundFile decode as OGG. "
        "The machine-readable manifest contains SHA-256 hashes, source path, expected/actual sizes, metadata, "
        "statuses, and sanitized errors.",
    ])
    if unexpected_final:
        lines.extend(["", "## Unexpected final files", "", *[f"- `{name}`" for name in unexpected_final]])
    if stop_reason:
        lines.extend(["", "## Acquisition stop reason", "", stop_reason])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=REPORT_PATH.parent,
                                     prefix=REPORT_PATH.name + ".", suffix=".tmp") as stream:
        temporary_name = stream.name
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_name, REPORT_PATH)


def run() -> int:
    frame = build_manifest()
    initial_final = {
        path.relative_to(FINAL_ROOT).as_posix()
        for path in FINAL_ROOT.rglob("*") if path.is_file()
    } if FINAL_ROOT.exists() else set()
    api = SafeKaggleApi()
    api.authenticate()  # Existing configured Kaggle credentials; never printed.
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_csv(frame, MANIFEST_PATH)
    stop_reason = ""

    for index, row in frame.iterrows():
        filename = str(row.filename)
        source_path = str(row.source_path)
        expected_size = int(row.expected_size_bytes)
        final = FINAL_ROOT / filename
        staged = STAGING_ROOT / source_path

        if final.exists():
            fields, error = _audio_fields(final, expected_size)
            if fields is not None:
                _mark_verified(frame, index, "skipped_verified", fields)
            else:
                frame.at[index, "download_status"] = "conflict"
                frame.at[index, "verification_status"] = "failed"
                frame.at[index, "actual_size_bytes"] = final.stat().st_size
                frame.at[index, "error_message"] = f"Existing destination preserved: {error}"
            _atomic_csv(frame, MANIFEST_PATH)
            continue

        request_succeeded = False
        protected_partial: Path | None = None
        try:
            if staged.exists():
                fields, error = _audio_fields(staged, expected_size)
                if fields is not None:
                    final.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(staged, final)  # Same-volume atomic promotion; never replaces a final path on Windows.
                    _mark_verified(frame, index, "resumed_staged_verified", fields)
                else:
                    resumable, marker_error = inspect_staged_partial(staged)
                    if not resumable:
                        frame.at[index, "download_status"] = "deferred"
                        frame.at[index, "verification_status"] = "deferred"
                        frame.at[index, "actual_size_bytes"] = staged.stat().st_size
                        frame.at[index, "error_message"] = marker_error or error
                        _atomic_csv(frame, MANIFEST_PATH)
                        continue
                    protected_partial = staged

            if frame.at[index, "verification_status"] != "verified" and frame.at[index, "verification_status"] != "deferred":
                api.begin_file(source_path)
                api.protect_partial(protected_partial)
                api.competition_download_file(
                    COMPETITION,
                    source_path,
                    path=str(STAGING_ROOT),
                    force=False,
                    quiet=True,
                    unzip=True,
                )
                api.protect_partial(None)
                request_succeeded = True
                if not staged.is_file():
                    raise FileNotFoundError("Kaggle returned without the expected staged OGG")
                fields, error = _audio_fields(staged, expected_size)
                if fields is None:
                    frame.at[index, "download_status"] = "failed"
                    frame.at[index, "verification_status"] = "failed"
                    frame.at[index, "actual_size_bytes"] = staged.stat().st_size
                    frame.at[index, "error_message"] = error
                else:
                    final.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(staged, final)
                    _mark_verified(frame, index, "downloaded", fields)
        except Exception as exc:
            api.protect_partial(None)
            frame.at[index, "download_status"] = "deferred" if isinstance(exc, UnsafePartialResumeError) else "failed"
            frame.at[index, "verification_status"] = "deferred" if isinstance(exc, UnsafePartialResumeError) else "failed"
            if staged.exists():
                frame.at[index, "actual_size_bytes"] = staged.stat().st_size
            frame.at[index, "error_message"] = safe_error(exc)
            if _is_stop_error(exc):
                stop_reason = f"Stopped at `{source_path}`: {safe_error(exc)}"
        _atomic_csv(frame, MANIFEST_PATH)
        verified = int(frame.verification_status.eq("verified").sum())
        remaining = int(frame.verification_status.ne("verified").sum())
        print(
            f"Progress {index + 1}/{len(frame)}; verified={verified}; pending/failed/deferred={remaining}; "
            f"current={source_path}; status={frame.at[index, 'verification_status']}",
            flush=True,
        )
        if request_succeeded:
            time.sleep(SUCCESS_REQUEST_DELAY_SECONDS)
        if stop_reason:
            break

    actual_final = {
        path.relative_to(FINAL_ROOT).as_posix()
        for path in FINAL_ROOT.rglob("*") if path.is_file()
    } if FINAL_ROOT.exists() else set()
    new_unapproved = sorted((actual_final - initial_final) - set(frame.filename))
    staged_count = sum(1 for path in STAGING_ROOT.rglob("*") if path.is_file()) if STAGING_ROOT.exists() else 0
    _write_report(frame, stop_reason, new_unapproved, staged_count)
    verified = int(frame.verification_status.eq("verified").sum())
    failed = int(frame.verification_status.isin(["failed", "deferred"]).sum())
    pending = int(frame.verification_status.eq("not_attempted").sum())
    print(
        f"M7 acquisition summary: approved={len(frame)}, downloaded={int(frame.download_status.eq('downloaded').sum())}, "
        f"verified={verified}, failed_or_deferred={failed}, pending={pending}, "
        f"unexpected_new_final={len(new_unapproved)}, staged_files={staged_count}",
        flush=True,
    )
    return 0 if verified == len(frame) and not new_unapproved and not stop_reason else 2


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        raise SystemExit(1)
