"""Download and verify the approved BirdCLEF+ 2026 audio subset.

Safe to rerun. It downloads one inventory-listed file at a time through the
authenticated Kaggle API, keeps resumable partials in a separate staging tree,
and promotes only size-matched, decodable OGG files into data/raw/train_audio.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
import email.utils
import random
import datetime as dt
import json
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
from kaggle.api.kaggle_api_extended import KaggleApi
import requests
import urllib3.exceptions as urllib3_exceptions
from tqdm import tqdm


MAX_FILE_ATTEMPTS = 5
MAX_FILE_WAIT_SECONDS = 30 * 60
BACKOFF_INITIAL_SECONDS = 60
BACKOFF_MAX_SECONDS = 15 * 60
SUCCESS_REQUEST_DELAY_SECONDS = 1.5
MANIFEST_REPLACE_RETRY_DELAYS_SECONDS = (0.1, 0.2, 0.4, 0.8)


class RetryBudgetExceeded(RuntimeError):
    """The per-file retry or waiting budget was exhausted safely."""


class UnsafePartialResumeError(RuntimeError):
    """A staged partial cannot be resumed without risking its existing bytes."""


def inspect_staged_partial(path: Path) -> tuple[bool, str]:
    """Preflight a raw staged file's Kaggle marker without changing either file."""
    marker_path = Path(f"{path}.kaggle-partial")
    if not path.is_file():
        return False, "Staged partial is absent"
    if not marker_path.is_file():
        return False, "Incomplete staged OGG has no Kaggle partial marker; preserved and deferred"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        validator = marker.get("validator")
        size = marker.get("size")
        partial_size = path.stat().st_size
        if not isinstance(validator, str) or not validator.strip():
            return False, "Kaggle partial marker has no valid identity token; staged OGG preserved and deferred"
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            return False, "Kaggle partial marker has no valid total size; staged OGG preserved and deferred"
        if not 0 < partial_size < size:
            return False, "Staged OGG size is not a resumable partial for its marker; preserved and deferred"
        return True, ""
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError) as exc:
        return False, f"Kaggle partial marker is invalid ({type(exc).__name__}); staged OGG preserved and deferred"


def validate_content_range(
    value: str | None,
    requested_offset: int,
    total_size: int | None,
    response_length: str | None = None,
) -> bool:
    """Validate a 206 Content-Range before any resumed bytes are appended."""
    if not value:
        return False
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", value.strip(), flags=re.IGNORECASE)
    if not match:
        return False
    start, end = int(match.group(1)), int(match.group(2))
    complete = match.group(3)
    if start != requested_offset or end < start:
        return False
    if response_length is not None:
        try:
            if int(response_length) != end - start + 1:
                return False
        except ValueError:
            return False
    if complete != "*":
        remote_size = int(complete)
        if end >= remote_size or (total_size is not None and remote_size != total_size):
            return False
    elif total_size is not None and end >= total_size:
        return False
    return True


def parse_retry_after(value: str | None, now: dt.datetime | None = None) -> float | None:
    """Parse Retry-After seconds or an HTTP date; return None if invalid."""
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
        return seconds if seconds >= 0 else None
    except ValueError:
        pass
    try:
        date = email.utils.parsedate_to_datetime(raw)
        if date is None:
            return None
        if date.tzinfo is None:
            date = date.replace(tzinfo=dt.timezone.utc)
        current = now or dt.datetime.now(dt.timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=dt.timezone.utc)
        return max(0.0, (date - current).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def fallback_backoff(attempt_number: int, uniform=random.uniform) -> float:
    """Return 50–100% jitter over 60s exponential windows, capped at 15m."""
    window = min(BACKOFF_INITIAL_SECONDS * (2 ** max(0, attempt_number - 1)), BACKOFF_MAX_SECONDS)
    return uniform(window * 0.5, float(window))


class SafeKaggleApi(KaggleApi):
    """KaggleApi with project-local bounded retry policy for file transfers."""

    def __init__(self):
        super().__init__()
        self._safe_attempts = 0
        self._safe_waited = 0.0
        self._safe_file = ""
        self._download_started = False
        self._protected_partial_path: Path | None = None

    def begin_file(self, file_name: str) -> None:
        self._safe_attempts = 0
        self._safe_waited = 0.0
        self._safe_file = file_name
        self._protected_partial_path = None

    def protect_partial(self, path: Path | None) -> None:
        self._protected_partial_path = path.resolve() if path is not None else None

    def _replace_with_validated_full_response(self, response, outfile: str, expected_size: int | None,
                                              chunk_size: int) -> int:
        """Stage a full 200 body separately; replace the old partial only after validation."""
        if expected_size is None:
            raise UnsafePartialResumeError(
                f"{self._safe_file}: cannot validate HTTP 200 full response without remote size; partial preserved"
            )
        header_length = response.headers.get("Content-Length")
        if header_length is not None:
            try:
                if int(header_length) != expected_size:
                    raise UnsafePartialResumeError(
                        f"{self._safe_file}: HTTP 200 Content-Length differs from remote size; partial preserved"
                    )
            except ValueError as exc:
                raise UnsafePartialResumeError(
                    f"{self._safe_file}: invalid HTTP 200 Content-Length; partial preserved"
                ) from exc

        output_path = Path(outfile)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", delete=False, dir=str(output_path.parent),
                prefix=f"{output_path.name}.", suffix=".full-response.tmp",
            ) as temporary:
                temp_path = temporary.name
                received = 0
                for data in response.iter_content(chunk_size):
                    if data:
                        temporary.write(data)
                        received += len(data)
                temporary.flush()

            if received != expected_size:
                raise UnsafePartialResumeError(
                    f"{self._safe_file}: HTTP 200 body has {received} bytes; expected "
                    f"{expected_size}; staged partial preserved"
                )
            if output_path.suffix.lower() == ".ogg":
                valid, error, _ = inspect_audio(Path(temp_path))
                if not valid:
                    raise UnsafePartialResumeError(
                        f"{self._safe_file}: HTTP 200 body failed OGG validation ({error}); staged partial preserved"
                    )

            # Same-directory replacement is atomic on the target filesystem.
            os.replace(temp_path, outfile)
            temp_path = None
            self._clear_resume_marker(outfile)
            return received
        finally:
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def reserve_attempt(self, status: int | None = None) -> None:
        if self._safe_attempts >= MAX_FILE_ATTEMPTS:
            raise RetryBudgetExceeded(
                f"{self._safe_file}: stopped after {MAX_FILE_ATTEMPTS} attempts"
            )
        self._safe_attempts += 1
        if status is None:
            print(
                f"{self._safe_file}: HTTP request attempt "
                f"{self._safe_attempts}/{MAX_FILE_ATTEMPTS}", flush=True
            )
        else:
            print(
                f"{self._safe_file}: retrying after HTTP {status}; request attempt "
                f"{self._safe_attempts}/{MAX_FILE_ATTEMPTS}", flush=True
            )

    def wait_before_retry(self, error: BaseException, attempt_number: int) -> None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        if status in (401, 403):
            kind = "authentication" if status == 401 else "permission/access"
            print(
                f"{self._safe_file}: HTTP {status} on attempt "
                f"{self._safe_attempts}/{MAX_FILE_ATTEMPTS}; stopping immediately", flush=True
            )
            raise RuntimeError(f"{self._safe_file}: HTTP {status} {kind} failure; stopping") from error
        retry_after = None
        if status == 429 and response is not None:
            retry_after = parse_retry_after(response.headers.get("Retry-After"))
        delay = retry_after if retry_after is not None else fallback_backoff(attempt_number)
        remaining = MAX_FILE_WAIT_SECONDS - self._safe_waited
        if delay > remaining:
            print(
                f"{self._safe_file}: HTTP {status or 'network'}, requested wait "
                f"{delay:.1f}s exceeds remaining {remaining:.1f}s; stopping",
                flush=True,
            )
            raise RetryBudgetExceeded(
                f"{self._safe_file}: requested wait {delay:.1f}s exceeds remaining "
                f"{remaining:.1f}s budget; stopping without early retry"
            ) from error
        self._safe_waited += delay
        print(
            f"{self._safe_file}: HTTP {status or 'network'}, retry delay {delay:.1f}s; "
            f"attempt {self._safe_attempts}/{MAX_FILE_ATTEMPTS}, retrying as "
            f"attempt {self._safe_attempts + 1}/{MAX_FILE_ATTEMPTS} "
            f"(wait budget {self._safe_waited:.1f}/{MAX_FILE_WAIT_SECONDS}s)",
            flush=True,
        )
        time.sleep(delay)

    def competition_download_file(self, competition, file_name, path=None, force=False, quiet=False, unzip=False):
        """Retry 429s from the Kaggle metadata/file URL request before streaming starts."""
        if not self._safe_file:
            self.begin_file(file_name)
        while True:
            self._download_started = False
            self.reserve_attempt()
            try:
                result = super().competition_download_file(
                    competition, file_name, path=path, force=force, quiet=quiet,
                    # Do not let SDK post-processing unzip over a protected raw partial.
                    unzip=False if self._protected_partial_path is not None else unzip,
                )
                if self._protected_partial_path is not None and not self._download_started:
                    raise UnsafePartialResumeError(
                        f"{file_name}: SDK skipped the protected staged partial; preserved and deferred"
                    )
                return result
            except Exception as exc:
                if self._download_started:
                    raise
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                if status in (401, 403):
                    raise RuntimeError(
                        f"{file_name}: HTTP {status} "
                        f"{'authentication' if status == 401 else 'permission/access'} failure; stopping"
                    ) from exc
                if status != 429:
                    raise
                if self._safe_attempts >= MAX_FILE_ATTEMPTS:
                    print(
                        f"{file_name}: HTTP 429 after "
                        f"{self._safe_attempts}/{MAX_FILE_ATTEMPTS} attempts; stopping",
                        flush=True,
                    )
                    raise RetryBudgetExceeded(
                        f"{file_name}: HTTP 429 after {self._safe_attempts} attempts; stopping"
                    ) from exc
                self.wait_before_retry(exc, self._safe_attempts)

    def download_file(self, response, outfile, http_client, quiet=True, resume=False,
                      chunk_size=1048576, max_retries=5, timeout=300):
        """Stream with Kaggle 2.2.5's identity/range resume logic and bounded retries.

        The transfer/marker logic below follows the installed Kaggle 2.2.5
        implementation. Only its retry branch is replaced by the project budget.
        """
        self._download_started = True
        outpath = os.path.dirname(outfile)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        content_length = response.headers.get("Content-Length")
        size = int(content_length) if content_length else None
        last_modified = response.headers.get("Last-Modified")
        if last_modified is None:
            remote_date = dt.datetime.now()
        else:
            remote_date = dt.datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z")
        remote_date_timestamp = time.mktime(remote_date.timetuple())
        resumable = response.headers.get("Accept-Ranges") == "bytes"
        resume_validator = response.headers.get("ETag") or last_modified
        retry_count = 0
        download_url = response.url
        original_method = response.request.method if hasattr(response, "request") else "GET"
        original_headers = dict(response.request.headers) if hasattr(response, "request") and hasattr(response.request, "headers") else {}

        protected_partial = self._protected_partial_path
        if protected_partial is not None and Path(outfile).resolve() != protected_partial:
            raise UnsafePartialResumeError(
                f"{self._safe_file}: Kaggle resolved a different staging path; existing staged OGG preserved"
            )
        if protected_partial is not None and not self._can_resume_partial(
            outfile, resume, resumable, resume_validator, size
        ):
            raise UnsafePartialResumeError(
                f"{self._safe_file}: staged OGG marker identity/size does not match remote file; preserved and deferred"
            )

        while True:
            try:
                file_exists = os.path.isfile(outfile)
                validated_range_bytes = None
                full_range_response = False
                resume_from_partial = retry_count == 0 and self._can_resume_partial(
                    outfile, resume, resumable, resume_validator, size
                )
                if retry_count > 0 or resume_from_partial:
                    size_read = os.path.getsize(outfile) if file_exists else 0
                    open_mode = "ab"
                    if size is not None and size_read == size:
                        os.utime(outfile, times=(remote_date_timestamp, remote_date_timestamp))
                        self._clear_resume_marker(outfile)
                        return
                    retry_headers = original_headers.copy()
                    retry_headers["Range"] = f"bytes={size_read}-"
                    if resume_validator is not None:
                        retry_headers["If-Range"] = resume_validator
                    if resume_from_partial:
                        self.reserve_attempt()
                    response = requests.request(
                        original_method, download_url, headers=retry_headers, stream=True, timeout=timeout
                    )
                    if response.status_code == 206:
                        content_range = response.headers.get("Content-Range")
                        if not validate_content_range(
                            content_range, size_read, size, response.headers.get("Content-Length")
                        ):
                            response.close()
                            raise UnsafePartialResumeError(
                                f"{self._safe_file}: invalid/mismatched Content-Range; partial and marker preserved"
                            )
                        range_match = re.fullmatch(
                            r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range.strip(), flags=re.IGNORECASE
                        )
                        validated_range_bytes = int(range_match.group(2)) - int(range_match.group(1)) + 1
                    elif response.status_code == 200:
                        size_read = 0
                        open_mode = "wb"
                        full_range_response = True
                    elif response.status_code == 416:
                        response.close()
                        self.reserve_attempt()
                        print(
                            f"{self._safe_file}: HTTP 416; sending full-file fallback as request "
                            f"{self._safe_attempts}/{MAX_FILE_ATTEMPTS}", flush=True
                        )
                        response = requests.request(
                            original_method, download_url, headers=original_headers, stream=True, timeout=timeout
                        )
                        response.raise_for_status()
                        size_read = 0
                        open_mode = "wb"
                        self._write_resume_marker(outfile, resume_validator, size)
                    else:
                        response.raise_for_status()
                        raise requests.exceptions.HTTPError(
                            f"Unexpected status code {response.status_code} when resuming download", response=response
                        )
                else:
                    size_read = 0
                    open_mode = "wb"
                    self._write_resume_marker(outfile, resume_validator, size)

                with tqdm(total=size, initial=size_read, unit="B", unit_scale=True,
                          unit_divisor=1024, disable=quiet) as pbar:
                    if full_range_response:
                        size_read = self._replace_with_validated_full_response(
                            response, outfile, size, chunk_size
                        )
                        pbar.update(size_read)
                    elif validated_range_bytes is not None:
                        # Buffer a resumed range first so malformed/truncated 206
                        # bodies cannot contaminate the existing partial.
                        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b") as buffered:
                            received = 0
                            for data in response.iter_content(chunk_size):
                                if data:
                                    buffered.write(data)
                                    received += len(data)
                            if received != validated_range_bytes:
                                raise UnsafePartialResumeError(
                                    f"{self._safe_file}: HTTP 206 body length mismatch; partial and marker preserved"
                                )
                            buffered.seek(0)
                            with open(outfile, "ab") as out:
                                while True:
                                    data = buffered.read(chunk_size)
                                    if not data:
                                        break
                                    out.write(data)
                                    out.flush()
                                    size_read += len(data)
                                    pbar.update(len(data))
                    else:
                        with open(outfile, open_mode) as out:
                            if type(response).__name__ == "HTTPResponse":
                                while True:
                                    data = response.read(chunk_size)
                                    if not data:
                                        break
                                    out.write(data)
                                    out.flush()
                                    size_read += len(data)
                                    pbar.update(len(data))
                            else:
                                for data in response.iter_content(chunk_size):
                                    if not data:
                                        break
                                    out.write(data)
                                    out.flush()
                                    size_read += len(data)
                                    pbar.update(len(data))
                os.utime(outfile, times=(remote_date_timestamp, remote_date_timestamp))
                if size is not None:
                    final_size = os.path.getsize(outfile)
                    if final_size != size:
                        raise ValueError(f"Downloaded file size ({final_size}) does not match expected size ({size})")
                self._clear_resume_marker(outfile)
                return
            except (
                requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError, requests.exceptions.HTTPError,
                urllib3_exceptions.ProtocolError, urllib3_exceptions.ReadTimeoutError, OSError,
            ) as exc:
                response_obj = getattr(exc, "response", None)
                status = getattr(response_obj, "status_code", None)
                if status in (401, 403):
                    kind = "authentication" if status == 401 else "permission/access"
                    print(
                        f"{self._safe_file}: HTTP {status} on attempt "
                        f"{self._safe_attempts}/{MAX_FILE_ATTEMPTS}; stopping immediately", flush=True
                    )
                    raise RuntimeError(f"{self._safe_file}: HTTP {status} {kind} failure; stopping") from exc
                # Kaggle 2.2.5 retries transfer HTTP/network errors generally;
                # keep that behavior, with 401/403 handled above and 429 using
                # Retry-After when it is valid.
                retry_count += 1
                if self._safe_attempts >= MAX_FILE_ATTEMPTS:
                    print(
                        f"{self._safe_file}: HTTP {status or 'network'} after "
                        f"{self._safe_attempts}/{MAX_FILE_ATTEMPTS} attempts; stopping",
                        flush=True,
                    )
                    raise RetryBudgetExceeded(
                        f"{self._safe_file}: stopped after {self._safe_attempts} transfer attempts"
                    ) from exc
                self.wait_before_retry(exc, retry_count)
                self.reserve_attempt()


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data/metadata/train.csv"
INVENTORY_CSV = ROOT / "data/metadata/birdclef_file_inventory.csv"
DEST_ROOT = ROOT / "data/raw/train_audio"
STAGING_ROOT = ROOT / "data/raw/.birdclef_2026_approved_staging"
MANIFEST_PATH = ROOT / "reports/ml/download_manifest.csv"
SUMMARY_PATH = ROOT / "reports/ml/download_summary.md"
COMPETITION = "birdclef-2026"
APPROVED_SPECIES = {
    "compot1", "swtman1", "rufhor2", "smbani", "saytan1", "crbthr1",
    "rutjac1", "rufnig1", "chobla1", "yecpar", "spispi1", "burowl",
}
EXCLUDED = {
    "swtman1/XC299145.ogg",
    "crbthr1/XC670199.ogg",
}
EXPECTED_COUNT = 3373
EXPECTED_BYTES = 1_130_140_362
COLUMNS = [
    "primary_label", "filename", "expected_size_bytes", "actual_size_bytes",
    "download_status", "verification_status", "sample_rate", "channels",
    "duration_seconds", "audio_format", "error_message",
]


def safe_error(exc: BaseException) -> str:
    """Avoid persisting signed URLs or credentials in the user-facing report."""
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(r"https?://\S+", "<URL redacted>", message)
    message = re.sub(r"(?i)(token|password|secret|authorization|key)=\S+", r"\1=<redacted>", message)
    message = re.sub(r"(?i)bearer\s+\S+", "Bearer <redacted>", message)
    return message[:1000]


def make_expected_manifest() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV, dtype={"primary_label": str, "filename": str})
    inventory = pd.read_csv(INVENTORY_CSV)
    inventory = inventory[inventory["FileName"].astype(str).str.startswith("train_audio/")]
    size_map = {
        str(row.FileName).removeprefix("train_audio/"): int(row.SizeBytes)
        for row in inventory.itertuples()
    }
    subset = train[train["primary_label"].isin(APPROVED_SPECIES)].copy()
    candidates = set(subset["filename"])
    approved = sorted((candidates & size_map.keys()) - EXCLUDED)
    unexplained = (candidates - size_map.keys()) - EXCLUDED
    excluded_present = EXCLUDED <= candidates
    if not excluded_present or unexplained:
        raise RuntimeError(f"Approved paths cannot be reconstructed exactly; unmatched={sorted(unexplained)}")
    if len(approved) != EXPECTED_COUNT or len(set(approved)) != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT} unique paths, found {len(approved)}")
    if sum(size_map[p] for p in approved) != EXPECTED_BYTES:
        raise RuntimeError("Inventory size sum differs from approved estimate; stopping before download")

    label_by_file = subset.drop_duplicates("filename").set_index("filename")["primary_label"].to_dict()
    rows = []
    for filename in approved:
        label = label_by_file[filename]
        if Path(filename).parts[0] != label or not filename.lower().endswith(".ogg"):
            raise RuntimeError(f"Unexpected path/label or format for {filename}")
        rows.append({
            "primary_label": label,
            "filename": filename,
            "expected_size_bytes": size_map[filename],
            "actual_size_bytes": pd.NA,
            "download_status": "not_attempted",
            "verification_status": "not_attempted",
            "sample_rate": pd.NA,
            "channels": pd.NA,
            "duration_seconds": pd.NA,
            "error_message": "",
        })
    frame = pd.DataFrame(rows, columns=COLUMNS)
    # Empty metadata fields otherwise infer float64 and reject later string values.
    for column in (
        "primary_label", "filename", "download_status", "verification_status",
        "audio_format", "error_message",
    ):
        frame[column] = frame[column].astype("object")
    return frame


def save_manifest(frame: pd.DataFrame) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    for attempt in range(len(MANIFEST_REPLACE_RETRY_DELAYS_SECONDS) + 1):
        try:
            os.replace(temporary, MANIFEST_PATH)
            return
        except PermissionError as exc:
            if attempt == len(MANIFEST_REPLACE_RETRY_DELAYS_SECONDS):
                raise PermissionError(
                    f"Unable to atomically replace manifest after {attempt + 1} attempts; "
                    f"complete temporary manifest preserved at {temporary}"
                ) from exc
            time.sleep(MANIFEST_REPLACE_RETRY_DELAYS_SECONDS[attempt])


def inspect_audio(path: Path) -> tuple[bool, str, dict]:
    """Decode audio incrementally; do not load the whole file into memory."""
    try:
        info = sf.info(str(path))
        if info.format != "OGG":
            return False, f"Unexpected decoded format: {info.format}", {}
        if info.samplerate <= 0 or info.channels <= 0 or info.frames <= 0:
            return False, "Invalid sample rate, channel count, or frame count", {}
        with sf.SoundFile(str(path), mode="r") as audio:
            while True:
                block = audio.read(262144, dtype="float32", always_2d=True)
                if len(block) == 0:
                    break
                if not np.isfinite(block).all():
                    return False, "Decoded audio contains non-finite samples", {}
        return True, "", {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_seconds": float(info.frames / info.samplerate),
            "audio_format": f"{info.format}/{info.subtype}",
        }
    except Exception as exc:  # SoundFile/libsndfile errors are recorded per file.
        return False, safe_error(exc), {}


def write_summary(frame: pd.DataFrame, start_files: set[str], stop_reason: str = "") -> None:
    actual_total = 0
    verified_count = 0
    for row in frame.itertuples():
        final = DEST_ROOT / row.filename
        if final.is_file():
            actual_total += final.stat().st_size
            if row.verification_status == "verified":
                verified_count += 1
    expected_total = int(frame["expected_size_bytes"].sum())
    missing = int((frame["verification_status"] != "verified").sum())
    downloaded = int(frame["download_status"].isin(["downloaded", "skipped_verified"]).sum())
    final_files = {
        p.relative_to(DEST_ROOT).as_posix()
        for p in DEST_ROOT.rglob("*") if p.is_file()
    } if DEST_ROOT.exists() else set()
    unapproved_final = final_files - set(frame.filename)
    new_unapproved_final = unapproved_final - start_files
    verified_rows = frame[frame.verification_status == "verified"]
    size_mismatches = int((verified_rows.actual_size_bytes != verified_rows.expected_size_bytes).sum())
    rates = sorted(int(x) for x in verified_rows.sample_rate.dropna().unique())
    channels = sorted(int(x) for x in verified_rows.channels.dropna().unique())
    formats = sorted(str(x) for x in verified_rows.audio_format.dropna().unique())
    duration_range = (
        f"{verified_rows.duration_seconds.min():.5f}–{verified_rows.duration_seconds.max():.3f} s"
        if not verified_rows.empty else "not available"
    )
    lines = [
        "# BirdCLEF+ 2026 Approved Audio Download Summary", "",
        f"- Approved files: {len(frame)}", f"- Successfully present/downloaded: {downloaded}",
        f"- Verified files: {verified_count}", f"- Failed, missing, or unverified: {missing}",
        f"- Total expected bytes: {expected_total:,}", f"- Total actual bytes at final destinations: {actual_total:,}",
        f"- New files created under train_audio by this run: {len(final_files - start_files)}",
        f"- Files outside the approved manifest currently present: {len(unapproved_final)}; new unapproved files created: {len(new_unapproved_final)}",
        f"- Size mismatches among verified rows: {size_mismatches}",
        "", "## Per-species progress", "", "| Primary label | Expected files | Inventory bytes | Downloaded/present | Verified | Failed/unverified |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, group in frame.groupby("primary_label", sort=True):
        lines.append(
            f"| `{label}` | {len(group)} | {int(group.expected_size_bytes.sum()):,} | "
            f"{int(group.download_status.isin(['downloaded','skipped_verified']).sum())} | "
            f"{int((group.verification_status == 'verified').sum())} | "
            f"{int((group.verification_status != 'verified').sum())} |"
        )
    lines.extend([
        "", "## Discrepancies and errors", "",
        "Rows with a non-empty `error_message` are recorded in `download_manifest.csv`. "
        "A file is counted verified only when its size matches the inventory and SoundFile decodes it as OGG.",
        f"Verified audio properties: sample rates {rates or 'none'} Hz; channel counts {channels or 'none'}; "
        f"formats {formats or 'none'}; duration range {duration_range}. "
        f"Size mismatches among verified files: {size_mismatches}.",
        "", "## Method and resume", "",
        "The workflow uses the Kaggle Python API bundled with Kaggle CLI 2.2.5 (version checked with "
        "`.venv\\Scripts\\kaggle.exe --version`): "
        "`KaggleApi.competition_download_file('birdclef-2026', 'train_audio/<label>/<filename>', "
        "path='data/raw/.birdclef_2026_approved_staging', unzip=True)`. This is a per-file request; "
        "the competition-wide archive endpoint is not used. Kaggle's downloader retries network failures "
        "and supports identity-checked byte-range resume. Each validated staged file is atomically moved "
        "into `data/raw/train_audio/<label>/<filename>` only if that destination does not already exist.",
        "Source paths and expected sizes are validated against the approved inventory before "
        "requests. Authentication details and per-file operational errors are not included in this summary.",
        "", "From the BioSonic project root, resume with:", "", "```powershell",
        ".\\.venv\\Scripts\\python.exe scripts\\download_approved_audio.py", "```",
        "The script reconstructs and validates the fixed approved list before any request, skips files "
        "that already exist and verify correctly, and continues from Kaggle's resumable staging partials. "
        "It never overwrites a final destination. Existing invalid destinations are recorded as conflicts.",
    ])
    if stop_reason:
        lines.extend(["", "## Workflow stop reason", "", stop_reason])
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    frame = make_expected_manifest()
    if MANIFEST_PATH.exists():
        prior = pd.read_csv(MANIFEST_PATH, dtype={"primary_label": str, "filename": str})
        if set(prior.columns) != set(COLUMNS) or set(prior.filename) != set(frame.filename):
            raise RuntimeError("Existing download manifest does not match the approved list; refusing to replace it")
        prior = prior.set_index("filename")
        frame = frame.set_index("filename")
        for column in COLUMNS:
            if column != "filename" and column in prior.columns:
                frame[column] = prior[column].reindex(frame.index).combine_first(frame[column])
        frame = frame.reset_index()[COLUMNS]

    expected_paths = set(frame.filename)
    initial_files = {
        p.relative_to(DEST_ROOT).as_posix()
        for p in DEST_ROOT.rglob("*") if p.is_file()
    } if DEST_ROOT.exists() else set()
    conflicts = []
    for rel in expected_paths:
        final = DEST_ROOT / rel
        if final.exists():
            conflicts.append(rel)
    if conflicts:
        print(f"Preexisting approved destination paths: {len(conflicts)}; each will be inspected and never overwritten.")

    api = SafeKaggleApi()
    api.authenticate()  # Uses configured authentication; never prints credentials.
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values("filename").reset_index(drop=True)
    save_manifest(frame)
    stopped = ""

    for index, row in frame.iterrows():
        rel = row["filename"]
        label = row["primary_label"]
        expected = int(row["expected_size_bytes"])
        request_succeeded = False
        final = DEST_ROOT / rel
        staged = STAGING_ROOT / "train_audio" / rel
        # A preexisting final is accepted only after size and full decode checks.
        if final.exists():
            actual = final.stat().st_size
            valid, err, audio = inspect_audio(final)
            if actual == expected and valid:
                frame.at[index, "actual_size_bytes"] = actual
                frame.at[index, "download_status"] = "skipped_verified"
                frame.at[index, "verification_status"] = "verified"
                frame.at[index, "error_message"] = ""
                for key, value in audio.items():
                    frame.at[index, key] = value
            else:
                frame.at[index, "actual_size_bytes"] = actual
                frame.at[index, "download_status"] = "conflict"
                frame.at[index, "verification_status"] = "failed"
                frame.at[index, "error_message"] = err or f"Existing file size {actual} != expected {expected}; not overwritten"
                save_manifest(frame)
                continue
        else:
            # Promote complete staged files. An incomplete OGG is only handed back
            # to Kaggle when its sidecar marker is structurally plausible; the API
            # override then checks that marker against the remote identity before
            # issuing a Range request or writing any bytes.
            need_download = not staged.exists()
            protected_partial = None
            if staged.exists():
                stage_size = staged.stat().st_size
                valid, err, audio = inspect_audio(staged)
                if stage_size == expected and valid:
                    final.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.rename(staged, final)  # Fails instead of replacing an existing destination.
                    except FileExistsError:
                        frame.at[index, "download_status"] = "conflict"
                        frame.at[index, "verification_status"] = "failed"
                        frame.at[index, "error_message"] = "Destination appeared during promotion; not overwritten"
                        save_manifest(frame)
                        continue
                    frame.at[index, "actual_size_bytes"] = final.stat().st_size
                    frame.at[index, "download_status"] = "downloaded"
                    frame.at[index, "verification_status"] = "verified"
                    frame.at[index, "error_message"] = ""
                    for key, value in audio.items():
                        frame.at[index, key] = value
                    need_download = False
                else:
                    resumable, marker_error = inspect_staged_partial(staged)
                    if resumable:
                        need_download = True
                        protected_partial = staged
                    else:
                        frame.at[index, "actual_size_bytes"] = stage_size
                        frame.at[index, "download_status"] = "deferred"
                        frame.at[index, "verification_status"] = "deferred"
                        frame.at[index, "error_message"] = marker_error or err or (
                            f"Staged size {stage_size} != expected {expected}; preserved and deferred"
                        )
                        save_manifest(frame)
                        continue
            if need_download:
                try:
                    source = f"train_audio/{rel}"
                    api.begin_file(source)
                    api.protect_partial(protected_partial)
                    api.competition_download_file(
                        COMPETITION, source, path=str(STAGING_ROOT), force=False, quiet=True, unzip=True
                    )
                    api.protect_partial(None)
                    request_succeeded = True
                    if not staged.is_file():
                        raise FileNotFoundError("Kaggle API returned without the expected staged OGG file")
                    actual = staged.stat().st_size
                    if actual != expected:
                        raise ValueError(f"Downloaded size {actual} != inventory size {expected}")
                    valid, err, audio = inspect_audio(staged)
                    if not valid:
                        raise ValueError(f"SoundFile verification failed: {err}")
                    final.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(staged, final)
                    frame.at[index, "actual_size_bytes"] = final.stat().st_size
                    frame.at[index, "download_status"] = "downloaded"
                    frame.at[index, "verification_status"] = "verified"
                    frame.at[index, "error_message"] = ""
                    for key, value in audio.items():
                        frame.at[index, key] = value
                except Exception as exc:
                    api.protect_partial(None)
                    frame.at[index, "actual_size_bytes"] = staged.stat().st_size if staged.exists() else pd.NA
                    deferred = isinstance(exc, UnsafePartialResumeError)
                    frame.at[index, "download_status"] = "deferred" if deferred else "failed"
                    frame.at[index, "verification_status"] = "deferred" if deferred else "failed"
                    frame.at[index, "error_message"] = safe_error(exc)
                    error_text = frame.at[index, "error_message"].lower()
                    if any(token in error_text for token in (
                        "name resolution", "failed to resolve", "connectionerror",
                        "connecttimeout", "readtimeout", "too many requests", "429",
                        "max retries exceeded", "service unavailable", "bad gateway",
                        "unauthorized", "401", "authentication required", "403", "permission/access",
                        "retrybudgetexceeded",
                    )):
                        stopped = (
                            f"Stopped after a persistent network/rate-limit/server error at {rel}: "
                            f"{frame.at[index, 'error_message']}. Remaining rows were left not_attempted; "
                            "rerun the same script to resume."
                        )
                        save_manifest(frame)
                        break
        save_manifest(frame)
        if (index + 1) % 25 == 0 or index == 0:
            verified = int((frame.verification_status == "verified").sum())
            print(f"Progress {index + 1}/{len(frame)}; verified={verified}; current={rel}", flush=True)
        if request_succeeded:
            time.sleep(SUCCESS_REQUEST_DELAY_SECONDS)

    # Ensure no unapproved file was added under the approved final subtree.
    final_files = {
        p.relative_to(DEST_ROOT).as_posix()
        for p in DEST_ROOT.rglob("*") if p.is_file()
    } if DEST_ROOT.exists() else set()
    added = final_files - initial_files
    unexpected = added - expected_paths
    if unexpected:
        stopped = "Unexpected files appeared under data/raw/train_audio/: " + ", ".join(sorted(unexpected)[:30])
    missing_final = expected_paths - final_files
    if missing_final:
        print(f"Missing final destinations: {len(missing_final)}", flush=True)
    write_summary(frame, initial_files, stopped)
    verified = int((frame.verification_status == "verified").sum())
    print(f"Finished: verified={verified}/{len(frame)}; missing={len(missing_final)}; unexpected={len(unexpected)}", flush=True)
    return 0 if verified == len(frame) and not unexpected else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        raise SystemExit(1)
