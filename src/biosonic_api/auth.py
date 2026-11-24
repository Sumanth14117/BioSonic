"""Authentication dependencies for HttpOnly cookie sessions."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from biosonic_api.auth_store import AuthStore


def get_auth_store(request: Request) -> AuthStore:
    return request.app.state.auth_store


def get_current_user(request: Request) -> dict[str, str]:
    settings = request.app.state.settings
    session_id = request.cookies.get(settings.auth_cookie_name)
    user = get_auth_store(request).lookup_session(session_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"code": "authentication_required", "message": "Sign in to continue."})
    return user


def validate_request_origin(request: Request) -> None:
    origin = request.headers.get("origin", "").rstrip("/")
    settings = request.app.state.settings
    if not origin or origin not in settings.auth_allowed_origins:
        raise HTTPException(status_code=403, detail={"code": "origin_not_allowed", "message": "This request origin is not allowed."})


def require_csrf(request: Request) -> None:
    validate_request_origin(request)
    token = request.headers.get("x-csrf-token")
    if not get_auth_store(request).consume_challenge_csrf(token):
        raise HTTPException(status_code=403, detail={"code": "csrf_failed", "message": "Refresh the page and try again."})
    return None


def get_authenticated_csrf_user(request: Request) -> dict[str, str]:
    user = get_current_user(request)
    validate_request_origin(request)
    settings = request.app.state.settings
    session_id = request.cookies.get(settings.auth_cookie_name)
    token = request.headers.get("x-csrf-token")
    if not get_auth_store(request).validate_session_csrf(token, session_id):
        raise HTTPException(status_code=403, detail={"code": "csrf_failed", "message": "Refresh the page and try again."})
    return user
