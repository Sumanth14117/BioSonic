"""Registration, login, cookie-session, and account endpoints."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from biosonic_api.auth import get_auth_store, get_authenticated_csrf_user, get_current_user, require_csrf, validate_request_origin
from biosonic_api.auth_store import AuthStore
from biosonic_api.config import Settings
from biosonic_api.schemas import AuthCredentials, AuthUser, CsrfResponse, EmailUpdateRequest, PasswordUpdateRequest, RegisterRequest

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _check_email(email: str) -> str:
    normalized = AuthStore.normalize_email(email)
    if len(normalized) > 254 or not EMAIL_PATTERN.fullmatch(normalized):
        raise HTTPException(status_code=422, detail={"code": "invalid_email", "message": "Enter a valid email address."})
    return normalized


def _issue_session(response: Response, store: AuthStore, settings: Settings, user: dict[str, str]) -> dict[str, object]:
    session_id, csrf_token = store.create_session(user["id"])
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=session_id,
        max_age=settings.auth_session_ttl_seconds,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="strict",
        path="/",
    )
    return {"user": user, "csrf_token": csrf_token}


@router.get("/csrf", response_model=CsrfResponse)
def csrf_token(request: Request, response: Response, store: AuthStore = Depends(get_auth_store)):
    settings = _settings(request)
    session_id = request.cookies.get(settings.auth_cookie_name)
    user = store.lookup_session(session_id)
    if user and session_id:
        token = store.rotate_csrf(session_id)
    else:
        token = store.create_csrf_challenge()
    return CsrfResponse(csrf_token=token, user=user)


@router.post("/register", response_model=CsrfResponse, status_code=201)
def register(
    payload: RegisterRequest,
    request: Request,
    response: Response,
    _: None = Depends(require_csrf),
    store: AuthStore = Depends(get_auth_store),
):
    validate_request_origin(request)
    email = _check_email(payload.email)
    if len(payload.password) < 12 or payload.password.isspace():
        raise HTTPException(status_code=422, detail={"code": "weak_password", "message": "Use a password with at least 12 characters that is not all whitespace."})
    user = store.create_user(email, payload.password)
    if user is None:
        raise HTTPException(status_code=409, detail={"code": "account_unavailable", "message": "An account could not be created with those details."})
    return CsrfResponse(**_issue_session(response, store, _settings(request), user))


@router.post("/login", response_model=CsrfResponse)
def login(
    payload: AuthCredentials,
    request: Request,
    response: Response,
    _: None = Depends(require_csrf),
    store: AuthStore = Depends(get_auth_store),
):
    validate_request_origin(request)
    user = store.authenticate(payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail={"code": "invalid_credentials", "message": "Email or password is incorrect."})
    return CsrfResponse(**_issue_session(response, store, _settings(request), user))


@router.get("/me", response_model=AuthUser)
def me(user: dict[str, str] = Depends(get_current_user)):
    return AuthUser(**user)


@router.post("/logout", status_code=204)
def logout(
    request: Request,
    response: Response,
    user: dict[str, str] = Depends(get_authenticated_csrf_user),
    store: AuthStore = Depends(get_auth_store),
):
    del user
    settings = _settings(request)
    store.revoke_session(request.cookies.get(settings.auth_cookie_name))
    response.delete_cookie(key=settings.auth_cookie_name, path="/", httponly=True, secure=settings.auth_cookie_secure, samesite="strict")
    response.status_code = 204
    return response


@router.patch("/me", response_model=AuthUser)
def update_email(
    payload: EmailUpdateRequest,
    user: dict[str, str] = Depends(get_authenticated_csrf_user),
    store: AuthStore = Depends(get_auth_store),
):
    email = _check_email(payload.email)
    if not store.update_email(user["id"], email):
        raise HTTPException(status_code=409, detail={"code": "email_unavailable", "message": "This email address cannot be used."})
    return AuthUser(id=user["id"], email=email)


@router.post("/me/password", status_code=204)
def change_password(
    payload: PasswordUpdateRequest,
    request: Request,
    response: Response,
    user: dict[str, str] = Depends(get_authenticated_csrf_user),
    store: AuthStore = Depends(get_auth_store),
):
    if len(payload.new_password) < 12 or payload.new_password.isspace():
        raise HTTPException(status_code=422, detail={"code": "weak_password", "message": "Use a password with at least 12 characters that is not all whitespace."})
    if store.authenticate(user["email"], payload.current_password) is None:
        raise HTTPException(status_code=400, detail={"code": "current_password_invalid", "message": "The current password is incorrect."})
    store.update_password(user["id"], payload.new_password)
    settings = _settings(request)
    response.delete_cookie(key=settings.auth_cookie_name, path="/", httponly=True, secure=settings.auth_cookie_secure, samesite="strict")
    response.status_code = 204
    return response
