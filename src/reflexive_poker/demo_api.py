from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fastapi import Cookie, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .demo_engine import DEFAULT_LLM_MODEL, DEMO_OPPONENT_TYPES
from .demo_service import (
    DemoConflictError,
    DemoNotFoundError,
    DemoPermissionError,
    DemoService,
)

OWNER_COOKIE = "poker_demo_owner"
DEFAULT_OPPONENTS = ("tag", "lag", "rock", "calling_station", "myopic")
DEMO_API_ERRORS = (DemoNotFoundError, DemoPermissionError, DemoConflictError, ValueError)


class CreateTableRequest(BaseModel):
    opponents: tuple[str, str, str, str, str] = DEFAULT_OPPONENTS
    provider_mode: Literal["mock", "live_aliyun"] = "mock"
    seed: int = Field(default=9200, ge=0, le=2_147_483_647)


class ActionRequest(BaseModel):
    action: Literal["fold", "check_call", "raise"]
    raise_scale: float = 0.5


class ControllerRequest(BaseModel):
    controller: Literal["human", "llm_closed_loop"]


class SeatControllerRequest(BaseModel):
    controller: Literal["rule_ai", "llm_closed_loop"]


class AdviceToggleRequest(BaseModel):
    enabled: bool


class SeatModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=80)


def _raise_http(exc: Exception) -> None:
    if isinstance(exc, DemoNotFoundError):
        raise HTTPException(status_code=404, detail="table_not_found") from exc
    if isinstance(exc, DemoPermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, DemoConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


def create_app(
    database: Path | None = None,
    *,
    live_call_limit: int | None = None,
    model_loader: Callable[[], tuple[str, ...]] | None = None,
) -> FastAPI:
    database = database or Path(
        os.getenv("POKER_DEMO_DB", ".local/poker-demo/poker_demo.sqlite3")
    )
    limit = live_call_limit or int(os.getenv("POKER_DEMO_LIVE_CALL_LIMIT", "200"))
    service = DemoService(
        database,
        live_call_limit=limit,
        **({"model_loader": model_loader} if model_loader is not None else {}),
    )
    app = FastAPI(title="Reflexive Poker Local Demo", version="0.1.0")
    app.state.demo_service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {
            "ok": True,
            "service": "reflexive-poker-demo",
            "defaultLiveModel": f"opencode-go/{DEFAULT_LLM_MODEL}",
            "liveCallLimit": service.live_call_limit,
        }

    @app.get("/api/strategies")
    async def strategies() -> dict[str, object]:
        return {
            "opponents": list(DEMO_OPPONENT_TYPES),
            "heroControllers": ["human", "llm_closed_loop"],
            "opponentControllers": ["rule_ai", "llm_closed_loop"],
            "defaultLiveModel": f"opencode-go/{DEFAULT_LLM_MODEL}",
        }

    @app.get("/api/models")
    async def models() -> dict[str, object]:
        return await service.model_catalog()

    @app.post("/api/tables")
    async def create_table(
        body: CreateTableRequest,
        response: Response,
        poker_demo_owner: str | None = Cookie(default=None),
    ) -> dict[str, object]:
        try:
            table, token = await service.create(
                owner_token=poker_demo_owner,
                opponents=body.opponents,
                provider_mode=body.provider_mode,
                seed=body.seed,
            )
        except DEMO_API_ERRORS as exc:
            _raise_http(exc)
        response.set_cookie(
            OWNER_COOKIE,
            token,
            httponly=True,
            samesite="strict",
            secure=False,
            max_age=60 * 60 * 24 * 30,
        )
        return service.snapshot(table, owner=True)

    @app.get("/api/tables/{table_id}")
    async def get_table(
        table_id: str,
        poker_demo_owner: str | None = Cookie(default=None),
    ) -> dict[str, object]:
        try:
            return await service.get(table_id, poker_demo_owner)
        except DEMO_API_ERRORS as exc:
            _raise_http(exc)

    @app.post("/api/tables/{table_id}/actions")
    async def act(
        table_id: str,
        body: ActionRequest,
        poker_demo_owner: str | None = Cookie(default=None),
    ) -> dict[str, object]:
        try:
            table = await service.action(
                table_id, poker_demo_owner, body.action, body.raise_scale
            )
            return service.snapshot(table, owner=True)
        except DEMO_API_ERRORS as exc:
            _raise_http(exc)

    @app.post("/api/tables/{table_id}/hero/controller")
    async def controller(
        table_id: str,
        body: ControllerRequest,
        poker_demo_owner: str | None = Cookie(default=None),
    ) -> dict[str, object]:
        try:
            table = await service.controller(table_id, poker_demo_owner, body.controller)
            return service.snapshot(table, owner=True)
        except DEMO_API_ERRORS as exc:
            _raise_http(exc)

    @app.post("/api/tables/{table_id}/hero/advice-toggle")
    async def advice_toggle(
        table_id: str,
        body: AdviceToggleRequest,
        poker_demo_owner: str | None = Cookie(default=None),
    ) -> dict[str, object]:
        try:
            table = await service.advice_enabled(table_id, poker_demo_owner, body.enabled)
            return service.snapshot(table, owner=True)
        except DEMO_API_ERRORS as exc:
            _raise_http(exc)

    @app.post("/api/tables/{table_id}/seats/{seat}/controller")
    async def seat_controller(
        table_id: str,
        seat: int,
        body: SeatControllerRequest,
        poker_demo_owner: str | None = Cookie(default=None),
    ) -> dict[str, object]:
        try:
            table = await service.seat_controller(
                table_id, poker_demo_owner, seat, body.controller
            )
            return service.snapshot(table, owner=True)
        except DEMO_API_ERRORS as exc:
            _raise_http(exc)

    @app.post("/api/tables/{table_id}/hero/advice")
    async def advice(
        table_id: str,
        poker_demo_owner: str | None = Cookie(default=None),
    ) -> dict[str, object]:
        try:
            table = await service.request_advice(table_id, poker_demo_owner)
            return service.snapshot(table, owner=True)
        except DEMO_API_ERRORS as exc:
            _raise_http(exc)

    @app.post("/api/tables/{table_id}/seats/{seat}/model")
    async def seat_model(
        table_id: str,
        seat: int,
        body: SeatModelRequest,
        poker_demo_owner: str | None = Cookie(default=None),
    ) -> dict[str, object]:
        try:
            table = await service.seat_model(table_id, poker_demo_owner, seat, body.model)
            return service.snapshot(table, owner=True)
        except DEMO_API_ERRORS as exc:
            _raise_http(exc)

    @app.post("/api/tables/{table_id}/next-hand")
    async def next_hand(
        table_id: str,
        poker_demo_owner: str | None = Cookie(default=None),
    ) -> dict[str, object]:
        try:
            table = await service.next_hand(table_id, poker_demo_owner)
            return service.snapshot(table, owner=True)
        except DEMO_API_ERRORS as exc:
            _raise_http(exc)

    @app.post("/api/tables/{table_id}/finish")
    async def finish(
        table_id: str,
        poker_demo_owner: str | None = Cookie(default=None),
    ) -> dict[str, object]:
        try:
            table = await service.finish(table_id, poker_demo_owner)
            return service.snapshot(table, owner=True)
        except DEMO_API_ERRORS as exc:
            _raise_http(exc)

    @app.websocket("/api/tables/{table_id}/events")
    async def events(websocket: WebSocket, table_id: str, after: int = 0) -> None:
        await websocket.accept()
        cursor = after
        try:
            while True:
                values = service.store.events_after(table_id, cursor)
                for event in values:
                    cursor = int(event["seq"])
                    await websocket.send_json(event)
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            return

    web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("reflexive_poker.demo_api:app", host="127.0.0.1", port=8790, reload=False)
