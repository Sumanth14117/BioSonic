"""Mock-only regression tests for the BioSonic Kaggle downloader."""

from __future__ import annotations

import datetime as dt
import email.utils
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import pandas as pd
import requests

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import download_approved_audio as downloader  # noqa: E402
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402


def http_error(status: int, headers: dict | None = None) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    return requests.HTTPError(f"HTTP {status}", response=response)


class FakeResponse:
    def __init__(self, status=200, body=b"abcdef", headers=None):
        self.status_code = status
        self.body = body
        self.headers = headers or {}
        self.url = "https://example.invalid/file.ogg"
        self.request = types.SimpleNamespace(method="GET", headers={"Accept": "audio/ogg"})
        self.closed = False

    def iter_content(self, chunk_size):
        if self.body:
            yield self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise http_error(self.status_code, self.headers)

    def close(self):
        self.closed = True


class FailingInitialResponse(FakeResponse):
    def iter_content(self, chunk_size):
        yield b"ab"
        raise requests.ConnectionError("mock interrupted stream")


class DownloaderRetryTests(unittest.TestCase):
    def _manifest_save_fixture(self, root: pathlib.Path):
        manifest = root / "download_manifest.csv"
        temporary = root / "download_manifest.csv.tmp"
        manifest.write_bytes(b"existing manifest\n")
        frame = pd.DataFrame({"filename": ["bird/example.ogg"], "status": ["verified"]})
        return manifest, temporary, frame

    def test_manifest_replace_retries_permission_error_then_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, temporary, frame = self._manifest_save_fixture(pathlib.Path(tmp))
            original_replace = downloader.os.replace
            calls = []

            def transient_then_success(source, destination):
                calls.append((source, destination))
                if len(calls) == 1:
                    raise PermissionError("transient sharing violation")
                return original_replace(source, destination)

            with mock.patch.object(downloader, "MANIFEST_PATH", manifest), mock.patch.object(
                downloader.os, "replace", side_effect=transient_then_success
            ), mock.patch.object(downloader.time, "sleep") as sleep, mock.patch.object(
                frame, "to_csv", wraps=frame.to_csv
            ) as write_csv:
                downloader.save_manifest(frame)

            self.assertEqual(len(calls), 2)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [0.1])
            write_csv.assert_called_once_with(temporary, index=False)
            self.assertFalse(temporary.exists())
            self.assertEqual(pd.read_csv(manifest).to_dict("records"), frame.to_dict("records"))

    def test_manifest_replace_retries_multiple_transient_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, temporary, frame = self._manifest_save_fixture(pathlib.Path(tmp))
            original_replace = downloader.os.replace
            calls = []

            def transient_twice_then_success(source, destination):
                calls.append((source, destination))
                if len(calls) < 3:
                    raise PermissionError("transient sharing violation")
                return original_replace(source, destination)

            with mock.patch.object(downloader, "MANIFEST_PATH", manifest), mock.patch.object(
                downloader.os, "replace", side_effect=transient_twice_then_success
            ), mock.patch.object(downloader.time, "sleep") as sleep, mock.patch.object(
                frame, "to_csv", wraps=frame.to_csv
            ) as write_csv:
                downloader.save_manifest(frame)

            self.assertEqual(len(calls), 3)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [0.1, 0.2])
            write_csv.assert_called_once_with(temporary, index=False)
            self.assertFalse(temporary.exists())

    def test_manifest_replace_exhaustion_preserves_main_and_complete_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, temporary, frame = self._manifest_save_fixture(pathlib.Path(tmp))
            original_main = manifest.read_bytes()
            with mock.patch.object(downloader, "MANIFEST_PATH", manifest), mock.patch.object(
                downloader.os, "replace", side_effect=PermissionError("persistent sharing violation")
            ) as replace, mock.patch.object(downloader.time, "sleep") as sleep, mock.patch.object(
                frame, "to_csv", wraps=frame.to_csv
            ) as write_csv:
                with self.assertRaisesRegex(PermissionError, "complete temporary manifest preserved"):
                    downloader.save_manifest(frame)

            self.assertEqual(replace.call_count, 5)
            self.assertEqual([c.args[0] for c in sleep.call_args_list], [0.1, 0.2, 0.4, 0.8])
            write_csv.assert_called_once_with(temporary, index=False)
            self.assertEqual(manifest.read_bytes(), original_main)
            self.assertTrue(temporary.is_file())
            self.assertEqual(pd.read_csv(temporary).to_dict("records"), frame.to_dict("records"))

    def test_manifest_replace_does_not_retry_non_permission_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, temporary, frame = self._manifest_save_fixture(pathlib.Path(tmp))
            original_main = manifest.read_bytes()
            with mock.patch.object(downloader, "MANIFEST_PATH", manifest), mock.patch.object(
                downloader.os, "replace", side_effect=OSError("disk error")
            ) as replace, mock.patch.object(downloader.time, "sleep") as sleep:
                with self.assertRaisesRegex(OSError, "disk error"):
                    downloader.save_manifest(frame)

            replace.assert_called_once_with(temporary, manifest)
            sleep.assert_not_called()
            self.assertEqual(manifest.read_bytes(), original_main)
            self.assertTrue(temporary.is_file())

    def test_summary_resume_instructions_are_project_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            frame = pd.DataFrame(columns=downloader.COLUMNS)
            with mock.patch.object(downloader, "DEST_ROOT", root / "audio"), mock.patch.object(
                downloader, "SUMMARY_PATH", root / "download_summary.md"
            ):
                downloader.write_summary(frame, set())

            summary = (root / "download_summary.md").read_text(encoding="utf-8")
            self.assertIn("From the BioSonic project root, resume with:", summary)
            self.assertIn(".\\.venv\\Scripts\\python.exe scripts\\download_approved_audio.py", summary)
            self.assertNotRegex(summary, r"(?i)\b[A-Z]:\\")

    def test_retry_after_parsing_and_backoff_budget(self):
        now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        self.assertEqual(downloader.parse_retry_after("12.5", now), 12.5)
        date = email.utils.format_datetime(now + dt.timedelta(seconds=45), usegmt=True)
        self.assertEqual(downloader.parse_retry_after(date, now), 45.0)
        self.assertIsNone(downloader.parse_retry_after(None, now))
        self.assertIsNone(downloader.parse_retry_after("invalid", now))
        for attempt, window in ((1, 60), (2, 120), (3, 240), (4, 480), (8, 900)):
            self.assertEqual(downloader.fallback_backoff(attempt, lambda low, high: low), window * 0.5)
            self.assertEqual(downloader.fallback_backoff(attempt, lambda low, high: high), float(window))

        api = downloader.SafeKaggleApi()
        api.begin_file("bird.ogg")
        api._safe_waited = 1790
        error = http_error(429, {"Retry-After": "11"})
        with mock.patch.object(downloader.time, "sleep") as sleep:
            with self.assertRaises(downloader.RetryBudgetExceeded):
                api.wait_before_retry(error, 1)
            sleep.assert_not_called()

    def test_api_401_403_stop_immediately_and_429_is_bounded(self):
        original = KaggleApi.competition_download_file
        try:
            for status in (401, 403):
                calls = []

                def denied(self, *args, _status=status, **kwargs):
                    calls.append(_status)
                    raise http_error(_status)

                KaggleApi.competition_download_file = denied
                api = downloader.SafeKaggleApi()
                api.begin_file("bird.ogg")
                with mock.patch.object(downloader.time, "sleep") as sleep:
                    with self.assertRaises(RuntimeError):
                        api.competition_download_file("competition", "bird.ogg")
                    self.assertEqual(calls, [status])
                    sleep.assert_not_called()

            calls = []

            def limited(self, *args, **kwargs):
                calls.append(1)
                raise http_error(429, {"Retry-After": "0"})

            KaggleApi.competition_download_file = limited
            api = downloader.SafeKaggleApi()
            api.begin_file("bird.ogg")
            with mock.patch.object(downloader.time, "sleep") as sleep:
                with self.assertRaises(downloader.RetryBudgetExceeded):
                    api.competition_download_file("competition", "bird.ogg")
                self.assertEqual(len(calls), downloader.MAX_FILE_ATTEMPTS)
                self.assertEqual(sleep.call_count, downloader.MAX_FILE_ATTEMPTS - 1)
        finally:
            KaggleApi.competition_download_file = original

    def test_416_fallback_consumes_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "sample.ogg"
            path.write_bytes(b"abc")
            pathlib.Path(f"{path}.kaggle-partial").write_text(
                json.dumps({"validator": "etag-1", "size": 6}), encoding="utf-8"
            )
            original = FakeResponse(
                headers={"Content-Length": "6", "Accept-Ranges": "bytes", "ETag": "etag-1"}
            )
            api = downloader.SafeKaggleApi()
            api.begin_file("bird/sample.ogg")
            api.reserve_attempt()  # Competition file request that supplied original.
            calls = [FakeResponse(416, b"", {}), FakeResponse(200, b"abcdef", {"Content-Length": "6"})]
            with mock.patch.object(downloader.requests, "request", side_effect=calls) as request:
                api.download_file(original, str(path), None, quiet=True, resume=True)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(api._safe_attempts, 3)  # API, Range request, 416 fallback.
            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertFalse(pathlib.Path(f"{path}.kaggle-partial").exists())

    def test_416_fallback_never_exceeds_five_total_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "sample.ogg"
            path.write_bytes(b"abc")
            pathlib.Path(f"{path}.kaggle-partial").write_text(
                json.dumps({"validator": "etag-1", "size": 6}), encoding="utf-8"
            )
            original = FailingInitialResponse(
                headers={"Content-Length": "6", "Accept-Ranges": "bytes", "ETag": "etag-1"}
            )
            api = downloader.SafeKaggleApi()
            api.begin_file("bird/sample.ogg")
            api.reserve_attempt()  # Initial API/file response.
            with mock.patch.object(
                downloader.requests, "request", side_effect=[FakeResponse(416, b"", {}) for _ in range(8)]
            ) as request, mock.patch.object(downloader, "fallback_backoff", return_value=0), mock.patch.object(
                downloader.time, "sleep"
            ):
                with self.assertRaises(downloader.RetryBudgetExceeded):
                    api.download_file(original, str(path), None, quiet=True, resume=True)
            self.assertEqual(api._safe_attempts, downloader.MAX_FILE_ATTEMPTS)
            # Four outgoing requests after the initial response: two ranges and two fallbacks.
            self.assertEqual(request.call_count + 1, downloader.MAX_FILE_ATTEMPTS)
            self.assertEqual(path.read_bytes(), b"abc")
            self.assertTrue(pathlib.Path(f"{path}.kaggle-partial").exists())

    def _resume_file(self, content_range: str):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = pathlib.Path(temp.name) / "sample.ogg"
        path.write_bytes(b"abc")
        marker = pathlib.Path(f"{path}.kaggle-partial")
        marker.write_text(json.dumps({"validator": "etag-1", "size": 6}), encoding="utf-8")
        original = FakeResponse(
            headers={"Content-Length": "6", "Accept-Ranges": "bytes", "ETag": "etag-1"}
        )
        resumed = FakeResponse(206, b"def", {"Content-Length": "3", "Content-Range": content_range})
        api = downloader.SafeKaggleApi()
        api.begin_file("bird/sample.ogg")
        api.reserve_attempt()
        api.protect_partial(path)
        request_patch = mock.patch.object(downloader.requests, "request", return_value=resumed)
        request = request_patch.start()
        self.addCleanup(request_patch.stop)
        return api, path, marker, original, resumed, request

    def test_valid_staged_file_and_marker_resume(self):
        api, path, marker, original, resumed, request = self._resume_file("bytes 3-5/6")
        api.download_file(original, str(path), None, quiet=True, resume=True)
        self.assertEqual(path.read_bytes(), b"abcdef")
        self.assertFalse(marker.exists())
        self.assertEqual(request.call_args.kwargs["headers"]["Range"], "bytes=3-")
        self.assertEqual(request.call_args.kwargs["headers"]["If-Range"], "etag-1")

    def test_206_with_wrong_content_range_preserves_partial(self):
        api, path, marker, original, resumed, request = self._resume_file("bytes 2-4/6")
        with self.assertRaises(downloader.UnsafePartialResumeError):
            api.download_file(original, str(path), None, quiet=True, resume=True)
        self.assertEqual(path.read_bytes(), b"abc")
        self.assertTrue(marker.exists())
        self.assertTrue(resumed.closed)

    def test_range_200_complete_body_atomically_replaces_partial(self):
        api, path, marker, original, _, _ = self._resume_file("bytes 3-5/6")
        full_response = FakeResponse(200, b"abcdef", {"Content-Length": "6"})

        def validate_temp(temp_path):
            self.assertEqual(path.read_bytes(), b"abc")
            self.assertEqual(pathlib.Path(temp_path).read_bytes(), b"abcdef")
            return True, "", {
                "sample_rate": 32000, "channels": 1, "duration_seconds": 1.0, "audio_format": "OGG/VORBIS"
            }

        with mock.patch.object(downloader.requests, "request", return_value=full_response), mock.patch.object(
            downloader, "inspect_audio", side_effect=validate_temp
        ):
            api.download_file(original, str(path), None, quiet=True, resume=True)
        self.assertEqual(path.read_bytes(), b"abcdef")
        self.assertFalse(marker.exists())

    def test_range_200_truncated_body_preserves_partial_and_marker(self):
        api, path, marker, original, _, _ = self._resume_file("bytes 3-5/6")
        truncated = FakeResponse(200, b"def", {"Content-Length": "6"})
        with mock.patch.object(downloader.requests, "request", return_value=truncated), mock.patch.object(
            downloader, "inspect_audio"
        ) as validate_audio:
            with self.assertRaises(downloader.UnsafePartialResumeError):
                api.download_file(original, str(path), None, quiet=True, resume=True)
            validate_audio.assert_not_called()
        self.assertEqual(path.read_bytes(), b"abc")
        self.assertTrue(marker.exists())

    def test_range_200_invalid_ogg_preserves_partial_and_marker(self):
        api, path, marker, original, _, _ = self._resume_file("bytes 3-5/6")
        invalid = FakeResponse(200, b"abcdef", {"Content-Length": "6"})
        with mock.patch.object(downloader.requests, "request", return_value=invalid), mock.patch.object(
            downloader, "inspect_audio", return_value=(False, "not an OGG", {})
        ):
            with self.assertRaises(downloader.UnsafePartialResumeError):
                api.download_file(original, str(path), None, quiet=True, resume=True)
        self.assertEqual(path.read_bytes(), b"abc")
        self.assertTrue(marker.exists())

    def test_resumed_transfer_401_403_stop_without_retry(self):
        for status in (401, 403):
            api, path, marker, original, _, request = self._resume_file("bytes 3-5/6")
            denied = FakeResponse(status, b"", {})
            with mock.patch.object(downloader.requests, "request", return_value=denied) as request, mock.patch.object(
                downloader.time, "sleep"
            ) as sleep:
                with self.assertRaises(RuntimeError):
                    api.download_file(original, str(path), None, quiet=True, resume=True)
                request.assert_called_once()
                sleep.assert_not_called()
            self.assertEqual(path.read_bytes(), b"abc")
            self.assertTrue(marker.exists())

    def test_resumed_transfer_429_uses_retry_after_and_stays_bounded(self):
        api, path, marker, original, _, _ = self._resume_file("bytes 3-5/6")
        limited = FakeResponse(429, b"", {"Retry-After": "0"})
        resumed = FakeResponse(206, b"def", {"Content-Length": "3", "Content-Range": "bytes 3-5/6"})
        with mock.patch.object(downloader.requests, "request", side_effect=[limited, resumed]) as request, mock.patch.object(
            downloader.time, "sleep"
        ) as sleep:
            api.download_file(original, str(path), None, quiet=True, resume=True)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(api._safe_attempts, 3)
        sleep.assert_called_once_with(0.0)
        self.assertEqual(path.read_bytes(), b"abcdef")
        self.assertFalse(marker.exists())

    def _configure_main(self, root: pathlib.Path, staged: bool, marker: dict | None = None):
        final_root = root / "final"
        stage_root = root / "stage"
        rel = "bird/sample.ogg"
        staged_path = stage_root / "train_audio" / rel
        final_path = final_root / rel
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if staged:
            staged_path.write_bytes(b"abc")
            if marker is not None:
                pathlib.Path(f"{staged_path}.kaggle-partial").write_text(json.dumps(marker), encoding="utf-8")
        row = {
            "primary_label": "bird", "filename": rel, "expected_size_bytes": 6,
            "actual_size_bytes": pd.NA, "download_status": "not_attempted",
            "verification_status": "not_attempted", "sample_rate": pd.NA,
            "channels": pd.NA, "duration_seconds": pd.NA, "audio_format": "", "error_message": "",
        }
        return final_root, stage_root, staged_path, final_path, pd.DataFrame([row], columns=downloader.COLUMNS)

    def _patch_main_paths(self, root, frame):
        patches = [
            mock.patch.object(downloader, "DEST_ROOT", root / "final"),
            mock.patch.object(downloader, "STAGING_ROOT", root / "stage"),
            mock.patch.object(downloader, "MANIFEST_PATH", root / "manifest.csv"),
            mock.patch.object(downloader, "SUMMARY_PATH", root / "summary.md"),
            mock.patch.object(downloader, "make_expected_manifest", return_value=frame),
            mock.patch.object(KaggleApi, "authenticate", return_value=None),
        ]
        return patches

    def test_staged_without_marker_is_preserved_and_deferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _, _, staged, _, frame = self._configure_main(root, staged=True)
            patches = self._patch_main_paths(root, frame)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], mock.patch.object(
                downloader, "inspect_audio", return_value=(False, "incomplete", {})
            ), mock.patch.object(downloader, "SafeKaggleApi") as api_class:
                self.assertEqual(downloader.main(), 2)
                api_class.return_value.competition_download_file.assert_not_called()
            self.assertEqual(staged.read_bytes(), b"abc")
            saved = pd.read_csv(root / "manifest.csv")
            self.assertEqual(saved.loc[0, "download_status"], "deferred")
            self.assertIn("no Kaggle partial marker", saved.loc[0, "error_message"])

    def test_mismatched_marker_is_preserved_and_deferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _, _, staged, _, frame = self._configure_main(
                root, staged=True, marker={"validator": "old-etag", "size": 6}
            )
            marker_path = pathlib.Path(f"{staged}.kaggle-partial")
            original = FakeResponse(headers={"Content-Length": "6", "Accept-Ranges": "bytes", "ETag": "new-etag"})

            def fake_base(self, competition, file_name, path=None, force=False, quiet=False, unzip=False):
                self.download_file(original, str(staged), None, quiet=True, resume=True)

            patches = self._patch_main_paths(root, frame)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], mock.patch.object(
                KaggleApi, "competition_download_file", fake_base
            ), mock.patch.object(downloader, "inspect_audio", return_value=(False, "incomplete", {})):
                self.assertEqual(downloader.main(), 2)
            self.assertEqual(staged.read_bytes(), b"abc")
            self.assertEqual(json.loads(marker_path.read_text(encoding="utf-8"))["validator"], "old-etag")
            saved = pd.read_csv(root / "manifest.csv")
            self.assertEqual(saved.loc[0, "download_status"], "deferred")
            self.assertIn("does not match remote file", saved.loc[0, "error_message"])

    def test_main_resumes_staged_ogg_with_matching_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _, _, staged, final, frame = self._configure_main(
                root, staged=True, marker={"validator": "etag-1", "size": 6}
            )
            original = FakeResponse(
                headers={"Content-Length": "6", "Accept-Ranges": "bytes", "ETag": "etag-1"}
            )
            resumed = FakeResponse(
                206, b"def", {"Content-Length": "3", "Content-Range": "bytes 3-5/6"}
            )

            def fake_base(self, competition, file_name, path=None, force=False, quiet=False, unzip=False):
                self.download_file(original, str(staged), None, quiet=True, resume=True)

            patches = self._patch_main_paths(root, frame)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], mock.patch.object(
                KaggleApi, "competition_download_file", fake_base
            ), mock.patch.object(downloader.requests, "request", return_value=resumed) as request, mock.patch.object(
                downloader, "inspect_audio", side_effect=lambda path: (
                    pathlib.Path(path).read_bytes() == b"abcdef", "incomplete", {
                        "sample_rate": 32000, "channels": 1, "duration_seconds": 1.0, "audio_format": "OGG/VORBIS"
                    } if pathlib.Path(path).read_bytes() == b"abcdef" else {}
                )
            ):
                self.assertEqual(downloader.main(), 0)
            self.assertFalse(staged.exists())
            self.assertEqual(final.read_bytes(), b"abcdef")
            self.assertEqual(request.call_args.kwargs["headers"]["Range"], "bytes=3-")
            self.assertEqual(request.call_args.kwargs["headers"]["If-Range"], "etag-1")

    def test_main_promotes_complete_range_200_only_after_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _, _, staged, final, frame = self._configure_main(
                root, staged=True, marker={"validator": "etag-1", "size": 6}
            )
            original = FakeResponse(
                headers={"Content-Length": "6", "Accept-Ranges": "bytes", "ETag": "etag-1"}
            )
            full = FakeResponse(200, b"abcdef", {"Content-Length": "6"})

            def fake_base(self, competition, file_name, path=None, force=False, quiet=False, unzip=False):
                self.download_file(original, str(staged), None, quiet=True, resume=True)

            def validate_audio(path):
                data = pathlib.Path(path).read_bytes()
                return (data == b"abcdef", "invalid", {
                    "sample_rate": 32000, "channels": 1, "duration_seconds": 1.0, "audio_format": "OGG/VORBIS"
                } if data == b"abcdef" else {})

            patches = self._patch_main_paths(root, frame)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], mock.patch.object(
                KaggleApi, "competition_download_file", fake_base
            ), mock.patch.object(downloader.requests, "request", return_value=full), mock.patch.object(
                downloader, "inspect_audio", side_effect=validate_audio
            ):
                self.assertEqual(downloader.main(), 0)
            self.assertEqual(final.read_bytes(), b"abcdef")
            self.assertFalse(staged.exists())
            self.assertFalse(pathlib.Path(f"{staged}.kaggle-partial").exists())

    def test_main_does_not_promote_incomplete_range_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _, _, staged, final, frame = self._configure_main(
                root, staged=True, marker={"validator": "etag-1", "size": 6}
            )
            marker = pathlib.Path(f"{staged}.kaggle-partial")
            original = FakeResponse(
                headers={"Content-Length": "6", "Accept-Ranges": "bytes", "ETag": "etag-1"}
            )
            truncated = FakeResponse(200, b"def", {"Content-Length": "6"})

            def fake_base(self, competition, file_name, path=None, force=False, quiet=False, unzip=False):
                self.download_file(original, str(staged), None, quiet=True, resume=True)

            patches = self._patch_main_paths(root, frame)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], mock.patch.object(
                KaggleApi, "competition_download_file", fake_base
            ), mock.patch.object(downloader.requests, "request", return_value=truncated), mock.patch.object(
                downloader, "inspect_audio", return_value=(False, "incomplete", {})
            ):
                self.assertEqual(downloader.main(), 2)
            self.assertFalse(final.exists())
            self.assertEqual(staged.read_bytes(), b"abc")
            self.assertTrue(marker.exists())

    def test_verified_final_file_remains_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            final_root, _, _, final, frame = self._configure_main(root, staged=False)
            final.parent.mkdir(parents=True, exist_ok=True)
            final.write_bytes(b"abcdef")
            patches = self._patch_main_paths(root, frame)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], mock.patch.object(
                downloader, "inspect_audio", return_value=(True, "", {
                    "sample_rate": 32000, "channels": 1, "duration_seconds": 1.0, "audio_format": "OGG/VORBIS"
                })
            ), mock.patch.object(downloader, "SafeKaggleApi") as api_class:
                self.assertEqual(downloader.main(), 0)
                api_class.return_value.competition_download_file.assert_not_called()
            self.assertEqual(final.read_bytes(), b"abcdef")
            saved = pd.read_csv(root / "manifest.csv")
            self.assertEqual(saved.loc[0, "download_status"], "skipped_verified")

    def test_successful_new_download_keeps_1500ms_delay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _, _, _, _, frame = self._configure_main(root, staged=False)
            staged = root / "stage" / "train_audio" / "bird" / "sample.ogg"

            def fake_base(self, competition, file_name, path=None, force=False, quiet=False, unzip=False):
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(b"abcdef")

            patches = self._patch_main_paths(root, frame)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], mock.patch.object(
                KaggleApi, "competition_download_file", fake_base
            ), mock.patch.object(downloader, "inspect_audio", return_value=(True, "", {
                "sample_rate": 32000, "channels": 1, "duration_seconds": 1.0, "audio_format": "OGG/VORBIS"
            })), mock.patch.object(downloader.time, "sleep") as sleep:
                self.assertEqual(downloader.main(), 0)
                sleep.assert_called_once_with(1.5)


if __name__ == "__main__":
    unittest.main()
