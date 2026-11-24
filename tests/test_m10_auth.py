"""M10.1 local authentication and account-isolation regression tests."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import pytest
import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biosonic_api.auth_store import AuthStore
from biosonic_api.config import Settings
from biosonic_api.main import create_app
from biosonic_api.services.m4_model import M4ModelService
from biosonic_api.services.registry import ModelRegistry

ORIGIN = "http://127.0.0.1:5173"
PASSWORD = "a secure forest passphrase 42"


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, values):
        return torch.arange(8, dtype=torch.float32, device=values.device).reshape(1, -1)


@pytest.fixture
def auth_client():
    with tempfile.TemporaryDirectory() as temporary:
        settings = Settings(auth_database_path=Path(temporary) / "auth.sqlite3", checkpoint_path=Path(temporary) / "unused.pt")
        service = M4ModelService(settings, model=FakeModel(), normalizer={"fit_partition": "internal_train", "mean": [0.0] * 128, "std": [1.0] * 128})
        app = create_app(settings, ModelRegistry(settings, m4_service=service))
        with TestClient(app) as client:
            yield client, app.state.auth_store


def csrf(client: TestClient) -> str:
    response = client.get("/api/v1/auth/csrf")
    assert response.status_code == 200
    return response.json()["csrf_token"]


def mutate_headers(token: str, origin: str = ORIGIN) -> dict[str, str]:
    return {"Origin": origin, "X-CSRF-Token": token}


def register(client: TestClient, email: str = "listener@example.org", password: str = PASSWORD):
    return client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password},
        headers=mutate_headers(csrf(client)),
    )


def test_registration_hashes_password_sets_httponly_session_and_supports_me(auth_client):
    client, store = auth_client
    response = register(client, " Listener@Example.org ")
    assert response.status_code == 201
    assert response.json()["user"]["email"] == "listener@example.org"
    assert len(response.json()["csrf_token"]) >= 32
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
    assert "biosonic_session" not in response.text
    assert client.get("/api/v1/auth/me").json()["email"] == "listener@example.org"
    with closing(sqlite3.connect(store.path)) as db:
        user_row = db.execute("SELECT password_hash FROM users").fetchone()
        session_hash = db.execute("SELECT session_hash, csrf_hash FROM sessions").fetchone()
    assert user_row[0] != PASSWORD and user_row[0].startswith("scrypt$")
    assert len(session_hash[0]) == 64 and response.cookies.get("biosonic_session") is not None


def test_registration_rejects_invalid_email_weak_password_and_duplicate(auth_client):
    client, _ = auth_client
    invalid_email = client.post("/api/v1/auth/register", json={"email": "not-an-email", "password": PASSWORD}, headers=mutate_headers(csrf(client)))
    assert invalid_email.status_code == 422
    weak = client.post("/api/v1/auth/register", json={"email": "weak@example.org", "password": "short"}, headers=mutate_headers(csrf(client)))
    assert weak.status_code == 422
    assert register(client).status_code == 201
    client.post("/api/v1/auth/logout", headers=mutate_headers(csrf(client)))
    duplicate = client.post("/api/v1/auth/register", json={"email": "LISTENER@example.org", "password": PASSWORD}, headers=mutate_headers(csrf(client)))
    assert duplicate.status_code == 409
    assert "already registered" not in duplicate.text.lower()


def test_login_uses_generic_errors_and_creates_a_new_session(auth_client):
    client, _ = auth_client
    assert register(client).status_code == 201
    client.post("/api/v1/auth/logout", headers=mutate_headers(csrf(client)))
    wrong = client.post("/api/v1/auth/login", json={"email": "listener@example.org", "password": "wrong passphrase"}, headers=mutate_headers(csrf(client)))
    unknown = client.post("/api/v1/auth/login", json={"email": "unknown@example.org", "password": "wrong passphrase"}, headers=mutate_headers(csrf(client)))
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    good = client.post("/api/v1/auth/login", json={"email": "listener@example.org", "password": PASSWORD}, headers=mutate_headers(csrf(client)))
    assert good.status_code == 200
    assert client.get("/api/v1/auth/me").status_code == 200


def test_login_prediction_and_history_are_account_scoped_and_logout_protects_routes(auth_client):
    client, _ = auth_client
    assert register(client, "owner@example.org").status_code == 201
    client.post("/api/v1/auth/logout", headers=mutate_headers(csrf(client)))
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "owner@example.org", "password": PASSWORD},
        headers=mutate_headers(csrf(client)),
    )
    assert login.status_code == 200

    buffer = io.BytesIO()
    waveform = 0.1 * np.sin(2 * np.pi * 440 * np.arange(32_000, dtype=np.float32) / 32_000)
    sf.write(buffer, waveform, 32_000, format="OGG", subtype="VORBIS")
    token = login.json()["csrf_token"]
    prediction = client.post(
        "/api/v1/predict",
        files={"file": ("login-flow.ogg", buffer.getvalue(), "audio/ogg")},
        headers=mutate_headers(token),
    )
    assert prediction.status_code == 200
    prediction_id = prediction.json()["prediction_id"]
    history = client.get("/api/v1/history")
    assert history.status_code == 200
    assert [item["id"] for item in history.json()["items"]] == [prediction_id]
    assert client.get(f"/api/v1/history/{prediction_id}").status_code == 200

    logout = client.post("/api/v1/auth/logout", headers=mutate_headers(csrf(client)))
    assert logout.status_code == 204
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/history").status_code == 401


def test_protected_prediction_and_history_require_authentication(auth_client):
    client, _ = auth_client
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/history").status_code == 401
    assert client.get("/api/v1/history/anything").status_code == 401
    response = client.post("/api/v1/predict", files={"file": ("bird.ogg", b"audio", "audio/ogg")})
    assert response.status_code == 401


def test_csrf_and_origin_are_required_for_cookie_mutations(auth_client):
    client, _ = auth_client
    assert register(client).status_code == 201
    no_token = client.post("/api/v1/auth/logout", headers={"Origin": ORIGIN})
    bad_origin = client.post("/api/v1/auth/logout", headers=mutate_headers(csrf(client), "https://attacker.example"))
    assert no_token.status_code == 403
    assert bad_origin.status_code == 403
    success = client.post("/api/v1/auth/logout", headers=mutate_headers(csrf(client)))
    assert success.status_code == 204


def test_logout_invalidates_session_and_clears_cookie(auth_client):
    client, _ = auth_client
    assert register(client).status_code == 201
    response = client.post("/api/v1/auth/logout", headers=mutate_headers(csrf(client)))
    assert response.status_code == 204
    assert 'biosonic_session="";' in response.headers["set-cookie"]
    assert client.get("/api/v1/auth/me").status_code == 401


def test_expired_sessions_are_rejected(auth_client):
    client, store = auth_client
    assert register(client).status_code == 201
    session_id = client.cookies.get("biosonic_session")
    with closing(sqlite3.connect(store.path)) as db:
        db.execute("UPDATE sessions SET expires_at = 0 WHERE session_hash = ?", (hashlib.sha256(session_id.encode()).hexdigest(),))
        db.commit()
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/history").status_code == 401


def test_session_csrf_tokens_are_session_bound_and_short_lived(auth_client):
    client, store = auth_client
    user = register(client).json()["user"]
    session_one = client.cookies.get("biosonic_session")
    token_one = client.get("/api/v1/auth/csrf").json()["csrf_token"]
    session_two, token_two = store.create_session(user["id"])
    assert store.validate_session_csrf(token_one, session_one)
    assert store.validate_session_csrf(token_two, session_two)
    assert not store.validate_session_csrf(token_two, session_one)
    with closing(sqlite3.connect(store.path)) as db:
        db.execute("UPDATE session_csrf_tokens SET expires_at = 0 WHERE token_hash = ?", (hashlib.sha256(token_one.encode()).hexdigest(),))
        db.commit()
    assert not store.validate_session_csrf(token_one, session_one)


def test_history_is_scoped_to_owner_and_audio_is_not_stored(auth_client):
    client, store = auth_client
    first = register(client, "one@example.org").json()["user"]["id"]
    record = {"model": "biosonic-m4-cnn", "model_version": "test", "prediction": "burowl", "confidence": 1.0, "probabilities": {"burowl": 1.0}, "audio": {"duration_seconds": 5, "sample_rate": 32000, "channels": 1, "sample_count": 160000, "format": "OGG/VORBIS"}}
    prediction_id = store.create_prediction(first, "field.ogg", 10, record)
    assert "audio_bytes" not in store.get_prediction(first, prediction_id)
    client.post("/api/v1/auth/logout", headers=mutate_headers(csrf(client)))
    assert register(client, "two@example.org").status_code == 201
    assert client.get("/api/v1/history").json()["items"] == []
    assert client.get(f"/api/v1/history/{prediction_id}").status_code == 404


def test_history_records_are_returned_to_their_owner(auth_client):
    client, store = auth_client
    user = register(client).json()["user"]["id"]
    prediction_id = store.create_prediction(user, "field.ogg", 12, {"prediction": "burowl", "probabilities": {"burowl": 1.0}})
    listing = client.get("/api/v1/history")
    assert listing.status_code == 200 and listing.json()["items"][0]["id"] == prediction_id
    detail = client.get(f"/api/v1/history/{prediction_id}")
    assert detail.status_code == 200 and detail.json()["file_name"] == "field.ogg"


def test_email_update_and_password_change_invalidate_sessions(auth_client):
    client, _ = auth_client
    assert register(client).status_code == 201
    changed = client.patch("/api/v1/auth/me", json={"email": "updated@example.org"}, headers=mutate_headers(csrf(client)))
    assert changed.status_code == 200 and changed.json()["email"] == "updated@example.org"
    password = client.post("/api/v1/auth/me/password", json={"current_password": PASSWORD, "new_password": "another secure passphrase 77"}, headers=mutate_headers(csrf(client)))
    assert password.status_code == 204
    assert client.get("/api/v1/auth/me").status_code == 401
    login = client.post("/api/v1/auth/login", json={"email": "updated@example.org", "password": "another secure passphrase 77"}, headers=mutate_headers(csrf(client)))
    assert login.status_code == 200


def test_schema_initialization_is_idempotent_and_csrf_tokens_are_hashed(auth_client):
    client, store = auth_client
    token = csrf(client)
    store.initialize()
    with closing(sqlite3.connect(store.path)) as db:
        stored = db.execute("SELECT token_hash FROM csrf_challenges").fetchone()[0]
        version = db.execute("SELECT version FROM schema_version").fetchone()[0]
    assert stored != token and len(stored) == 64
    assert version == 1
