from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi.testclient import TestClient

from reflexive_poker.demo_api import create_app
from reflexive_poker.demo_llm import LLM_TIMEOUT_SECONDS
from reflexive_poker.demo_llm import decide as real_decide
from reflexive_poker.demo_service import DemoService


def test_demo_api_persists_owner_table_and_keeps_spectators_read_only(tmp_path: Path) -> None:
    database = tmp_path / "demo.sqlite3"
    app = create_app(database)
    with TestClient(app) as owner:
        created = owner.post("/api/tables", json={})
        assert created.status_code == 200
        state = created.json()
        table_id = state["tableId"]
        assert state["owner"] is True
        assert state["canAct"] is True

        toggle = owner.post(
            f"/api/tables/{table_id}/hero/advice-toggle", json={"enabled": True}
        )
        assert toggle.status_code == 200
        advised = owner.post(f"/api/tables/{table_id}/hero/advice")
        assert advised.status_code == 200
        assert advised.json()["lastAdvice"]["readOnly"] is True
        assert advised.json()["strategy"]["version"] == 1

        cookies = dict(owner.cookies)

    restarted = create_app(database)
    with TestClient(restarted, cookies=cookies) as restored:
        state = restored.get(f"/api/tables/{table_id}").json()
        assert state["owner"] is True
        assert state["lastAdvice"] is not None

    with TestClient(restarted) as spectator:
        state = spectator.get(f"/api/tables/{table_id}").json()
        assert state["owner"] is False
        assert state["providerUsage"] == {}
        denied = spectator.post(
            f"/api/tables/{table_id}/actions",
            json={"action": "check_call", "raise_scale": 0.5},
        )
        assert denied.status_code == 403


def test_mock_llm_controls_hand_and_applies_post_hand_patch(tmp_path: Path) -> None:
    app = create_app(tmp_path / "demo.sqlite3")
    with TestClient(app) as client:
        table_id = client.post("/api/tables", json={"provider_mode": "mock"}).json()[
            "tableId"
        ]
        response = client.post(
            f"/api/tables/{table_id}/hero/controller",
            json={"controller": "llm_closed_loop"},
        )
        assert response.status_code == 200
        state = response.json()
        assert state["phase"] == "hand_complete"
        assert state["controller"] == "llm_closed_loop"
        assert state["strategy"]["version"] == 2
        assert state["providerUsage"]["mock_calls"] >= 2


def test_switch_to_human_invalidates_in_flight_llm_response(
    tmp_path: Path, monkeypatch
) -> None:
    service = DemoService(tmp_path / "demo.sqlite3")

    def slow_decide(table, actor=None):
        time.sleep(0.08)
        return real_decide(table, actor)

    monkeypatch.setattr("reflexive_poker.demo_service.decide", slow_decide)

    async def scenario() -> None:
        table, token = await service.create(
            owner_token=None,
            opponents=("tag", "lag", "rock", "calling_station", "myopic"),
            provider_mode="mock",
            seed=9200,
        )
        delegating = asyncio.create_task(
            service.controller(table.table_id, token, "llm_closed_loop")
        )
        await asyncio.sleep(0.02)
        await service.controller(table.table_id, token, "human")
        await delegating
        state = await service.get(table.table_id, token)
        assert state["controller"] == "human"
        assert state["canAct"] is True
        assert not any(
            item["seat"] == 0 for item in state["hand"]["actions"]
        )

    asyncio.run(scenario())


def test_other_agent_can_switch_to_llm_and_take_its_action(tmp_path: Path) -> None:
    app = create_app(tmp_path / "demo.sqlite3")
    with TestClient(app) as client:
        state = client.post("/api/tables", json={}).json()
        table_id = state["tableId"]
        switched = client.post(
            f"/api/tables/{table_id}/seats/1/controller",
            json={"controller": "llm_closed_loop"},
        )
        assert switched.status_code == 200
        assert switched.json()["hand"]["seats"][1]["controller"] == "llm_closed_loop"

        acted = client.post(
            f"/api/tables/{table_id}/actions",
            json={"action": "check_call", "raise_scale": 0.5},
        )
        assert acted.status_code == 200
        seat_one_actions = [
            item for item in acted.json()["hand"]["actions"] if item["seat"] == 1
        ]
        assert seat_one_actions
        assert seat_one_actions[0]["controller"] == "llm_closed_loop"


def test_llm_timeout_is_sixty_seconds() -> None:
    assert LLM_TIMEOUT_SECONDS == 60


def test_llm_failure_does_not_switch_back_to_human(tmp_path: Path, monkeypatch) -> None:
    service = DemoService(tmp_path / "demo.sqlite3")

    def failing_decide(table, actor=None):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("reflexive_poker.demo_service.decide", failing_decide)

    async def scenario() -> None:
        table, token = await service.create(
            owner_token=None,
            opponents=("tag", "lag", "rock", "calling_station", "myopic"),
            provider_mode="mock",
            seed=9200,
        )
        await service.controller(table.table_id, token, "llm_closed_loop")
        state = await service.get(table.table_id, token)
        assert state["controller"] == "llm_closed_loop"
        assert state["phase"] == "waiting_llm"
        assert state["pausedReason"].startswith("action_provider_failure_mock")

    asyncio.run(scenario())
