"""Small SQLite user, session, CSRF, and prediction-history store."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


PASSWORD_SCRYPT_N = 1 << 14
PASSWORD_SCRYPT_R = 8
PASSWORD_SCRYPT_P = 1
PASSWORD_SCRYPT_LENGTH = 32


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=PASSWORD_SCRYPT_N,
        r=PASSWORD_SCRYPT_R, p=PASSWORD_SCRYPT_P, dklen=PASSWORD_SCRYPT_LENGTH,
    )
    return f"scrypt${PASSWORD_SCRYPT_N}${PASSWORD_SCRYPT_R}${PASSWORD_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_hex, expected_hex = encoded.split("$", 5)
        if algorithm != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=len(bytes.fromhex(expected_hex)),
        )
        return hmac.compare_digest(digest.hex(), expected_hex)
    except (ValueError, TypeError, MemoryError):
        return False


class AuthStore:
    """Versioned, non-destructive SQLite persistence for local API accounts."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path, session_ttl_seconds: int = 43_200):
        self.path = Path(path)
        self.session_ttl_seconds = session_ttl_seconds

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = db.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            if row and row["version"] > self.SCHEMA_VERSION:
                raise RuntimeError("Authentication database schema is newer than this application supports.")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    csrf_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    revoked_at INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at);
                CREATE TABLE IF NOT EXISTS session_csrf_tokens (
                    session_hash TEXT NOT NULL REFERENCES sessions(session_hash) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    PRIMARY KEY(session_hash, token_hash)
                );
                CREATE TABLE IF NOT EXISTS csrf_challenges (
                    token_hash TEXT PRIMARY KEY,
                    expires_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL CHECK (file_size >= 0),
                    response_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_user_created ON predictions(user_id, created_at DESC);
                """
            )
            if row is None:
                db.execute("INSERT INTO schema_version(version) VALUES (?)", (self.SCHEMA_VERSION,))
            else:
                db.execute("UPDATE schema_version SET version = ?", (self.SCHEMA_VERSION,))

    @staticmethod
    def normalize_email(email: str) -> str:
        return email.strip().casefold()

    def create_user(self, email: str, password: str) -> dict[str, str] | None:
        now = int(time.time())
        user_id = str(uuid.uuid4())
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO users(id, email, password_hash, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, self.normalize_email(email), hash_password(password), now, now),
                )
        except sqlite3.IntegrityError:
            return None
        return {"id": user_id, "email": self.normalize_email(email)}

    def authenticate(self, email: str, password: str) -> dict[str, str] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, email, password_hash FROM users WHERE email = ? COLLATE NOCASE",
                (self.normalize_email(email),),
            ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return {"id": row["id"], "email": row["email"]}

    def user_by_id(self, user_id: str) -> dict[str, str] | None:
        with self._connect() as db:
            row = db.execute("SELECT id, email FROM users WHERE id = ?", (user_id,)).fetchone()
        return {"id": row["id"], "email": row["email"]} if row else None

    def update_email(self, user_id: str, email: str) -> bool:
        try:
            with self._connect() as db:
                cursor = db.execute(
                    "UPDATE users SET email = ?, updated_at = ? WHERE id = ?",
                    (self.normalize_email(email), int(time.time()), user_id),
                )
            return cursor.rowcount == 1
        except sqlite3.IntegrityError:
            return False

    def update_password(self, user_id: str, password: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (hash_password(password), int(time.time()), user_id),
            )
            db.execute("UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL", (int(time.time()), user_id))

    def create_session(self, user_id: str) -> tuple[str, str]:
        session_id = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        now = int(time.time())
        with self._connect() as db:
            db.execute(
                "INSERT INTO sessions(session_hash, user_id, csrf_hash, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                (_token_hash(session_id), user_id, _token_hash(csrf_token), now, now + self.session_ttl_seconds),
            )
            db.execute(
                "INSERT INTO session_csrf_tokens(session_hash, token_hash, expires_at) VALUES (?, ?, ?)",
                (_token_hash(session_id), _token_hash(csrf_token), min(now + 900, now + self.session_ttl_seconds)),
            )
        return session_id, csrf_token

    def lookup_session(self, session_id: str | None) -> dict[str, str] | None:
        if not session_id:
            return None
        with self._connect() as db:
            row = db.execute(
                "SELECT u.id, u.email FROM sessions s JOIN users u ON u.id = s.user_id "
                "WHERE s.session_hash = ? AND s.revoked_at IS NULL AND s.expires_at > ?",
                (_token_hash(session_id), int(time.time())),
            ).fetchone()
        return {"id": row["id"], "email": row["email"]} if row else None

    def revoke_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._connect() as db:
            db.execute("UPDATE sessions SET revoked_at = ? WHERE session_hash = ? AND revoked_at IS NULL", (int(time.time()), _token_hash(session_id)))

    def rotate_csrf(self, session_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        session_hash = _token_hash(session_id)
        with self._connect() as db:
            db.execute("DELETE FROM session_csrf_tokens WHERE expires_at <= ?", (now,))
            db.execute(
                "UPDATE sessions SET csrf_hash = ? WHERE session_hash = ? AND revoked_at IS NULL AND expires_at > ?",
                (_token_hash(token), session_hash, now),
            )
            db.execute(
                "INSERT INTO session_csrf_tokens(session_hash, token_hash, expires_at) "
                "SELECT ?, ?, MIN(expires_at, ?) FROM sessions WHERE session_hash = ? AND revoked_at IS NULL AND expires_at > ?",
                (session_hash, _token_hash(token), now + 900, session_hash, now),
            )
        return token

    def create_csrf_challenge(self, ttl_seconds: int = 900) -> str:
        token = secrets.token_urlsafe(32)
        with self._connect() as db:
            db.execute("DELETE FROM csrf_challenges WHERE expires_at <= ?", (int(time.time()),))
            db.execute("INSERT INTO csrf_challenges(token_hash, expires_at) VALUES (?, ?)", (_token_hash(token), int(time.time()) + ttl_seconds))
        return token

    def consume_challenge_csrf(self, token: str | None) -> bool:
        if not token:
            return False
        digest = _token_hash(token)
        with self._connect() as db:
            cursor = db.execute("DELETE FROM csrf_challenges WHERE token_hash = ? AND expires_at > ?", (digest, int(time.time())))
        return cursor.rowcount == 1

    def validate_session_csrf(self, token: str | None, session_id: str | None) -> bool:
        if not token or not session_id:
            return False
        now = int(time.time())
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM sessions s JOIN session_csrf_tokens c ON c.session_hash = s.session_hash "
                "WHERE s.session_hash = ? AND c.token_hash = ? AND c.expires_at > ? "
                "AND s.revoked_at IS NULL AND s.expires_at > ?",
                (_token_hash(session_id), _token_hash(token), now, now),
            ).fetchone()
        return row is not None

    def create_prediction(self, user_id: str, file_name: str, file_size: int, response: dict[str, Any]) -> str:
        prediction_id = str(uuid.uuid4())
        stored_response = {**response, "prediction_id": prediction_id}
        with self._connect() as db:
            db.execute(
                "INSERT INTO predictions(id, user_id, file_name, file_size, response_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (prediction_id, user_id, file_name, file_size, json.dumps(stored_response, allow_nan=False), int(time.time())),
            )
        return prediction_id

    @staticmethod
    def _prediction_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "file_name": row["file_name"], "file_size": row["file_size"],
            "created_at": row["created_at"], "response": json.loads(row["response_json"]),
        }

    def list_predictions(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, file_name, file_size, created_at, response_json FROM predictions WHERE user_id = ? ORDER BY created_at DESC, id LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [self._prediction_record(row) for row in rows]

    def get_prediction(self, user_id: str, prediction_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, file_name, file_size, created_at, response_json FROM predictions WHERE id = ? AND user_id = ?",
                (prediction_id, user_id),
            ).fetchone()
        return self._prediction_record(row) if row else None
