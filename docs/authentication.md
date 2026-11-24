# M10.1 Authentication and Account Storage

M10.1 adds local account authentication to the existing FastAPI and React application. The API remains a local research service; this implementation has not been hardened or deployed for public internet use.

## Storage

The API uses Python's standard-library `sqlite3` module. On startup it creates `data/auth/biosonic_auth.sqlite3` if needed and applies additive `CREATE TABLE IF NOT EXISTS` initialization tracked by a schema version. It does not drop tables or replace an existing database. Set `BIOSONIC_AUTH_DATABASE` to select another path. The project ignores `data/`, so the local account database is not tracked by Git.

Tables store normalized email addresses, scrypt password hashes, session hash/expiry/revocation data, short-lived hashed CSRF tokens, and prediction response metadata. Prediction history contains the submitted safe filename, size, model output and audio metadata; the uploaded audio bytes are not stored.

## Passwords and sessions

- Registration requires a syntactically valid email and a password between 12 and 128 characters.
- Passwords are hashed with salted scrypt (`N=16384`, `r=8`, `p=1`, 32-byte output). Plaintext passwords are never written to SQLite or browser storage.
- Login failures use the same response for unknown email and incorrect password.
- A successful login creates a 256-bit random opaque session identifier. Only its SHA-256 digest is stored in SQLite.
- The browser receives the identifier only in an `HttpOnly`, `SameSite=Strict`, path-root cookie with a 12-hour expiry. Cookies are not persisted in localStorage. `Secure` is off for the documented localhost HTTP workflow and can be enabled with `BIOSONIC_AUTH_COOKIE_SECURE=true` behind HTTPS.
- Logout revokes the server-side session and clears the cookie. Password changes revoke all of that account's sessions.

## CSRF and origin checks

The frontend fetches a CSRF token into in-memory React state. Anonymous registration/login challenges are random, stored as hashes, expire after 15 minutes, and are consumed by a successful CSRF check. Authenticated session CSRF tokens are random, stored as hashes tied to a session, and expire after 15 minutes; overlapping tokens allow multiple open tabs to finish safely. State-changing browser routes require the configured `Origin` and `X-CSRF-Token` header. The default allowlist is `http://127.0.0.1:5173,http://localhost:5173`; configure `BIOSONIC_ALLOWED_ORIGINS` as a comma-separated allowlist when the local frontend origin changes. The Vite proxy keeps the documented setup same-origin from the browser.

## API routes

| Method | Route | Access | Purpose |
|---|---|---|---|
| GET | `/api/v1/auth/csrf` | Public | Restore a session and obtain an in-memory CSRF token. |
| POST | `/api/v1/auth/register` | CSRF challenge | Create an account and start a session. |
| POST | `/api/v1/auth/login` | CSRF challenge | Authenticate and start a session. |
| POST | `/api/v1/auth/logout` | Session + session CSRF | Revoke the session. |
| GET | `/api/v1/auth/me` | Session | Read the current account profile. |
| PATCH | `/api/v1/auth/me` | Session + session CSRF | Change the account email. |
| POST | `/api/v1/auth/me/password` | Session + session CSRF | Change password and revoke sessions. |
| GET | `/api/v1/history` | Session | List at most 100 predictions owned by this user. |
| GET | `/api/v1/history/{prediction_id}` | Session | Fetch one prediction owned by this user. |
| POST | `/api/v1/predict` | Session + session CSRF | Run the existing M4 inference and save its summary to the caller's history. |

Prediction response inference and M4 preprocessing remain the same; the response adds a `prediction_id` to identify the saved account record. Ownership is checked in SQL for every history lookup; an ID from another account returns the same not-found response as an unknown ID. Health and model inventory routes remain public for operations checks.

## Browser history behavior

M9's former local-only history is not imported or assigned to an account. It is left untouched in the browser and is not shown in the account history. New successful analyses are saved on the server against the authenticated user. The history pages store no audio bytes.

## Local configuration

The defaults are suitable only for the local Vite proxy. Relevant optional settings:

```powershell
$env:BIOSONIC_AUTH_DATABASE = 'data/auth/biosonic_auth.sqlite3'
$env:BIOSONIC_ALLOWED_ORIGINS = 'http://127.0.0.1:5173'
$env:BIOSONIC_AUTH_COOKIE_SECURE = 'false' # Use true behind HTTPS.
```

From the project root, start the API with `.\.venv\Scripts\python.exe scripts\run_m8_api.py`. In another PowerShell window, run `npm run dev` from `frontend/`. See [the frontend guide](../frontend/README.md) and [API usage](api/usage.md).

## Limitations before any hosted deployment

This local milestone does not include email verification, password recovery, login throttling/lockout, account deletion, database encryption-at-rest, automated backups, multi-process deployment review, or a production TLS/reverse-proxy configuration. Add those controls and perform a dedicated security review before exposing user accounts to a network. Local SQLite files contain sensitive credential hashes and prediction metadata; protect and back them up according to the deployment's data policy.

## M10.1 verification

In the existing main environment, the authentication and M8 regression suites passed **23 tests**. They cover registration validation and duplicate handling, generic login failures, session expiry/revocation, CSRF and Origin enforcement, account changes, database initialization, owner-scoped history, and a synthetic-audio flow that logs in, predicts, retrieves history, logs out, and confirms protected access is denied. Frontend tests passed **14 tests**, including auth-provider state transitions, route guards, API request behavior, history and navigation. Oxlint, TypeScript compilation, and the Vite production build passed. One Starlette deprecation warning remains for its `httpx`-based TestClient. Browser-driven authentication review was not performed; this is automated component/API verification only.

From the repository root, start the backend with `.\.venv\Scripts\python.exe scripts\run_m8_api.py`; in a second PowerShell window run `cd frontend` and `npm run dev`. Run checks with `.\.venv\Scripts\python.exe -m pytest tests\test_m10_auth.py tests\test_m8_api.py -v`, then run `npm run test`, `npm run lint`, and `npm run build` from `frontend/`.
