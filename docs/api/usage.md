# BioSonic M8 Local API Usage

The API exposes the existing M4 recording-level CNN. It does not include a rejection class for unknown species, so every valid input is assigned among its trained labels. It is a local research API, not a deployed or production-ready wildlife detector.

## Environment and startup

Use the existing main `.venv` (Python 3.14.7, PyTorch CUDA build). HTTP verification used FastAPI 0.165.0, Uvicorn 0.54.0, and python-multipart 0.0.34. These packages are installed in the existing `.venv`; `.venv-yamnet` is not used by the API. The dependency ranges are listed in `requirements-api.txt`.

The current `.venv` already has the verified packages; do not reinstall them. If creating a separate compatible environment later, the optional dependency ranges are in `requirements-api.txt`.

From the repository root, start the API:

```powershell
.\.venv\Scripts\python.exe scripts\run_m8_api.py
```

Local base URL: `http://127.0.0.1:8000`<br>
Interactive OpenAPI documentation: `http://127.0.0.1:8000/docs`

The runner was verified to start Uvicorn on `127.0.0.1:8000` and load the M4 checkpoint on the RTX 4060 Laptop GPU. `/`, `/health`, `/api/v1/health`, `/api/v1/models`, and `/docs` returned HTTP 200. The generated OpenAPI schema includes the documented route set.

The service loads `models/m4_custom_cnn/best_model.pt` once at application startup. Set `BIOSONIC_M4_CHECKPOINT` to a compatible local checkpoint if the default location differs. `BIOSONIC_DEVICE` accepts `auto`, `cpu`, or `cuda`; `auto` selects CUDA when available. `BIOSONIC_MAX_UPLOAD_BYTES` defaults to 26,214,400 (25 MiB); duration defaults to 1–180 seconds.

## Health and model inventory

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/models
```

`/health` is process liveness. `/api/v1/health` reports model readiness. M7 appears as disabled in the model inventory because it remains an experimental research study and is not integrated as a reliable detector. No M7 prediction endpoint is exposed.

## Sign in and upload a recording

Prediction and history routes require an authenticated account. The browser app obtains a CSRF token, keeps it in memory, and sends it with state-changing requests; the API sets the session in an HttpOnly cookie. For the verified local workflow, use the React frontend at `http://127.0.0.1:5173` to register or sign in, then submit recordings from Audio Analysis. Do not place session IDs or passwords in browser local storage or scripts committed to the repository.

The authentication API also supports `GET /api/v1/auth/csrf`, `POST /api/v1/auth/register`, `POST /api/v1/auth/login`, `GET /api/v1/auth/me`, `POST /api/v1/auth/logout`, `PATCH /api/v1/auth/me`, and `POST /api/v1/auth/me/password`. Authenticated history is available at `GET /api/v1/history` and `GET /api/v1/history/{prediction_id}`. State-changing requests require an allowed `Origin` and `X-CSRF-Token`; see [the authentication guide](../authentication.md) for the cookie, CSRF, database, and configuration details.

## Upload a recording

The API request must include the session cookie and a valid `X-CSRF-Token` obtained through the authentication flow. The frontend manages both automatically. An unauthenticated request returns `401`; a missing or invalid CSRF token is rejected. Do not use the former unauthenticated `curl` example.

Successful responses contain a recording-level prediction, confidence, the configured class probabilities, model name/version, and decoded audio metadata. Segment probabilities are softmaxed per five-second segment and averaged following M4 evaluation behavior.

Use the authenticated frontend to upload an audio file that you are authorized to process. A successful response follows this shape; values below are placeholders, not a model result:

```json
{
  "prediction_id": "<prediction-id>",
  "model": "<model-id>",
  "model_version": "<model-version>",
  "prediction": "<configured-label>",
  "confidence": 0.0,
  "probabilities": {"<configured-label>": 0.0},
  "audio": {
    "duration_seconds": 0.0,
    "sample_rate": 32000,
    "channels": 1,
    "sample_count": 0,
    "format": "OGG/VORBIS"
  }
}
```

## Input contract

- Filename extension `.ogg`; file content must actually decode as OGG/Vorbis.
- Mono, 32 kHz audio. The API rejects different sample rates or channel counts; it does not resample or downmix.
- Duration 1–180 seconds and file size at most the configured upload limit.
- Uploads are copied to a random system temporary filename, fully decoded/validated, and deleted after the request. The user-supplied filename is never used as a path and is not returned or logged.
- Short final five-second windows are right-zero-padded, the same policy used by M4. Invalid padded Mel frames are set to zero after applying the training-only per-band normalizer.

Common error responses use `{ "detail": { "code": "...", "message": "..." } }`. Typical status codes are 400 (invalid upload/filename), 413 (size limit), 415 (unsupported format/extension), 422 (audio dimensions/duration/decoding), 503 (model unavailable), and 500 (inference failure). Internal paths and stack traces are not returned.

Live negative-upload checks returned `415` for a `.wav` filename, `422` for corrupt OGG content, and `413` for an oversized multipart upload. The upload temporary-file set returned to its initial state after successful prediction and after these failed requests.

The M8 test module passed all 10 tests, including its HTTP integration tests. The downloader and M3–M8 main-environment regression selection passed 114 tests with no failures or skips. Running `pytest tests` as one collection in `.venv` stops at the M5 module because `resampy` is intentionally part of the separate YAMNet environment; the M5/M7 suites passed there separately (22 tests). Pytest emitted a Starlette deprecation warning about its `httpx`-based `TestClient`; it did not affect HTTP checks. The API remains a local research service, not a deployment or production-readiness claim.

## Model limitations

M4 was an interim recording-level experiment using a previously evaluated holdout. It does not detect multiple species in a recording, reject out-of-scope species, or estimate animal population. Audio activity summaries must not be interpreted as population estimates.

## M10.3 browser signal analysis

The Audio Analysis page also decodes the selected file locally for waveform, exploratory STFT/Mel, spectrum, and signal-level summaries. These plots are computed in a Web Worker and are separate from backend prediction. The UI labels sample rate, channel count, and duration as browser-decoded metadata; source bit depth is not available. The visual STFT is 1,024-point symmetric Hann with 256-sample hop and complete frames, showing relative power dB. Its Slaney-normalized Mel view uses 128 bands up to `min(16 kHz, Nyquist)`. This is not a claim about M4's internal feature input; M4 remains authoritative and validates OGG/Vorbis contents, mono channel, 32 kHz rate, duration, and maximum upload size.

Low-energy frames use an explicitly exploratory threshold (100 ms frame, 50 ms hop, RMS `<0.005`). Peak thresholds are diagnostic only. These acoustic measurements do not establish species presence, abundance, or population.

## M10.4 batch requests

Use the authenticated **Batch analysis** page in the React app to select or drop several files. The client validates each `.ogg` extension, nonempty content, and the unchanged 25 MiB per-file limit. The backend performs authoritative content validation for every independent request. At most two requests run concurrently; each uses the existing session cookie, CSRF header, and `/api/v1/predict` route. There is no separate batch inference endpoint or batch database table.

The queue and exported CSV/JSON reflect only the current browser selection. Successful entries contain the actual API response; failed/cancelled entries remain visibly distinct. CSV uses proper quote escaping. Cancellation aborts the browser request and queued items but cannot guarantee already received server work stops. Prediction summaries also appear in authenticated account history. Batch class counts describe model outputs for that selection and are not ecological population estimates.
