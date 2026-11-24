"""Authenticated, owner-scoped prediction history."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from biosonic_api.auth import get_auth_store, get_current_user
from biosonic_api.auth_store import AuthStore

router = APIRouter(prefix="/api/v1/history", tags=["history"])


@router.get("")
def list_history(user: dict[str, str] = Depends(get_current_user), store: AuthStore = Depends(get_auth_store)):
    return {"items": store.list_predictions(user["id"])}


@router.get("/{prediction_id}")
def get_history_item(
    prediction_id: str,
    user: dict[str, str] = Depends(get_current_user),
    store: AuthStore = Depends(get_auth_store),
):
    item = store.get_prediction(user["id"], prediction_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "history_not_found", "message": "This prediction was not found."})
    return item
