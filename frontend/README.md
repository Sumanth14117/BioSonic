# BioSonic Frontend

React 19 + TypeScript interface for the local BioSonic FastAPI service. M10.1 supplies cookie-based authentication and M10.2 adds account-aware dashboards and analytics.

## Requirements

- Node.js 22.12 or newer (verified here with Node 22.22.2).
- npm 10 or newer (verified here with npm 10.9.7).
- The local BioSonic Python `.venv` with API dependencies installed and a compatible M4 checkpoint. The environment and checkpoint are local-only prerequisites; neither is included in Git.
- Recharts 3.11.0 is used for history-derived charts; Motion is not installed. Short transitions and reduced-motion behavior are CSS-based.
- A fresh-clone installation has not been verified.

## Run locally

From the repository root, start the backend in one PowerShell window:

```powershell
.\.venv\Scripts\python.exe scripts\run_m8_api.py
```

In another PowerShell window:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite server proxies `/api/*` and `/health` to `http://127.0.0.1:8000`, avoiding development CORS changes to the Python backend.

## Environment configuration

Copy `.env.example` to `.env.local`. Leave `VITE_API_BASE_URL` blank to use the Vite development proxy. If set, it must be the API origin, for example `http://127.0.0.1:8000` (do not add a trailing slash). Direct cross-origin production use requires an approved backend CORS origin or a same-origin reverse proxy; the M8 backend is not configured for arbitrary browser origins.

Vite reads `VITE_API_BASE_URL` at build time. Do not place credentials or secret values in `VITE_*` variables; browser variables are public.

## Pages and API integration

- **Overview** queries `/api/v1/health` and `/api/v1/models`; new prediction history is stored per authenticated account on the API.
- **Dashboard analytics** show counts and charts derived from the authenticated account's most recent 100 saved prediction summaries. Activity is grouped by UTC day; class mix uses the returned top label; duration bands use decoded response metadata. Empty history renders empty states, not sample values.
- **Audio analysis** accepts `.ogg` files up to 25 MiB, previews audio locally, and sends multipart field `file` to `POST /api/v1/predict`. A separate Web Worker computes a downsampled waveform, exploratory STFT/Mel views, spectrum, and signal measurements from browser-decoded samples; these are not the M4 model's input features.
- **Batch analysis** validates a multi-file selection against the same per-file contract and queues authenticated calls to the existing prediction endpoint with at most two requests in flight. It shows real per-file responses and current-batch-only comparisons and CSV/JSON exports. It adds no persistence table.
- **Prediction results** shows the API’s predicted code, confidence, audio metadata, and ranked probabilities returned by the active model. The frontend does not add species common names or ecological interpretation.
- **Authentication and account** provide registration, login, logout, profile email changes, and password changes. Protected routes are backed by API authorization, not only client-side route guards.
- **Model information** summarizes experiment methods and limitations without publishing static dataset-derived scores. Only M4 is exposed by the active API; M7 is experimental/research-only and disabled.
- **About BioSonic** describes the current pipeline and scientific limits.

The API remains a closed-set classifier over its fixed trained labels. Authenticated history stores response metadata and the supplied filename in the local SQLite database, but not audio bytes. Earlier browser-only history is left untouched and is not imported into an account. No inference is simulated. Cancelling a browser batch request cannot guarantee already received server-side inference stops.

The results view cannot replay a previous upload because the API deliberately does not retain source audio. To listen again, choose the original recording locally. Displayed model confidence is not calibrated evidence of biological presence, and dashboard counts are not wildlife population estimates.

## Design and responsive behavior

The application keeps a charcoal and forest-green foundation with raised data panels, restrained green/amber accents, clear empty/loading/error states, and accessible focus styles. The desktop sidebar becomes a dismissible mobile drawer; dashboard charts collapse to one column and auth, upload, and result panels reflow at narrow widths. CSS supports the reduced-motion preference. Page/card entrances, orbit artwork, toasts, upload state, Recharts bars/areas, and confidence ring use short CSS/Recharts animations; no extra animation dependency was added.

Dashboard aggregation lives in `src/lib/analytics.ts`; it operates only on the current user's API history response. The API returns at most 100 recent records, so dashboard totals and distributions describe that available window rather than a lifetime count. Local signal metrics use RMS `sqrt(sum(x²)/N)`, mean `sum(x)/N`, and absolute peak; low-energy uses 100 ms frames, 50 ms hop, and exploratory RMS `<0.005`. The visual STFT uses NFFT 1024, hop 256, Hann, relative power dB; its 128-band Slaney Mel view is exploratory, not a model feature cache. Acoustic outputs do not establish population abundance. Browser-driven desktop/tablet/mobile visual review remains outstanding.

## Quality commands

```powershell
npm test -- --run
npm run lint
npm run build
```

The Vite production output is written to `frontend/dist/`, which is ignored by Git. From the repository root, start FastAPI with `& .\.venv\Scripts\python.exe scripts/run_m8_api.py`, then start the UI with `cd frontend`, `npm ci`, and `npm run dev`. `npm run lint` invokes Oxlint (not ESLint). A fresh-clone installation has not been verified; the API also needs the local M4 checkpoint before it can serve predictions.
