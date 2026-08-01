from __future__ import annotations

import asyncio
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Any

from .demo_engine import DemoConfig, DemoTable
from .demo_llm import LLM_TIMEOUT_SECONDS, decide, reflect_and_patch
from .demo_store import DemoStore, owner_hash


class DemoNotFoundError(Exception):
    pass


class DemoPermissionError(Exception):
    pass


class DemoConflictError(Exception):
    pass


class DemoService:
    def __init__(self, database: Path, *, live_call_limit: int = 200) -> None:
        self.store = DemoStore(database)
        self.live_call_limit = live_call_limit
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._driver_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def create(
        self,
        *,
        owner_token: str | None,
        opponents: tuple[str, ...],
        provider_mode: str,
        seed: int,
    ) -> tuple[DemoTable, str]:
        token = owner_token or secrets.token_urlsafe(32)
        token_hash = owner_hash(token)
        active = self.store.active_for_owner(token_hash)
        if active is not None:
            return active, token
        table = DemoTable(
            DemoConfig(seed=seed, opponents=opponents, provider_mode=provider_mode)
        )
        self.store.save(table, token_hash)
        return table, token

    def _load(self, table_id: str) -> tuple[DemoTable, str]:
        loaded = self.store.load(table_id)
        if loaded is None:
            raise DemoNotFoundError(table_id)
        return loaded

    @staticmethod
    def _require_owner(stored_hash: str, token: str | None) -> str:
        if token is None or not secrets.compare_digest(stored_hash, owner_hash(token)):
            raise DemoPermissionError("owner_cookie_required")
        return stored_hash

    def snapshot(self, table: DemoTable, *, owner: bool) -> dict[str, Any]:
        value = table.snapshot(owner=owner)
        live_calls = int(table.provider_usage["live_calls"])
        value["liveCallBudget"] = {
            "used": live_calls,
            "limit": self.live_call_limit,
            "warning": live_calls >= int(self.live_call_limit * 0.8),
            "exhausted": live_calls >= self.live_call_limit,
        }
        value["owner"] = owner
        return value

    async def get(self, table_id: str, token: str | None) -> dict[str, Any]:
        async with self._locks[table_id]:
            table, stored_hash = self._load(table_id)
            is_owner = bool(
                token and secrets.compare_digest(stored_hash, owner_hash(token))
            )
            return self.snapshot(table, owner=is_owner)

    async def _mutate(self, table_id: str, token: str | None, operation) -> DemoTable:
        async with self._locks[table_id]:
            table, stored_hash = self._load(table_id)
            self._require_owner(stored_hash, token)
            try:
                operation(table)
            except ValueError as exc:
                raise DemoConflictError(str(exc)) from exc
            self.store.save(table, stored_hash)
            return table

    async def action(
        self, table_id: str, token: str | None, action: str, raise_scale: float
    ) -> DemoTable:
        table = await self._mutate(
            table_id,
            token,
            lambda table: table.apply_action(table.hero_seat, action, raise_scale),
        )
        await self.drive_llm(table_id)
        table, _ = self._load(table_id)
        return table

    async def controller(
        self, table_id: str, token: str | None, controller: str
    ) -> DemoTable:
        table = await self._mutate(
            table_id, token, lambda value: value.set_controller(controller)
        )
        if controller == "llm_closed_loop":
            await self.drive_llm(table_id)
            table, _ = self._load(table_id)
        return table

    async def seat_controller(
        self, table_id: str, token: str | None, seat: int, controller: str
    ) -> DemoTable:
        table = await self._mutate(
            table_id, token, lambda value: value.set_seat_controller(seat, controller)
        )
        await self.drive_llm(table_id)
        table, _ = self._load(table_id)
        return table

    async def advice_enabled(
        self, table_id: str, token: str | None, enabled: bool
    ) -> DemoTable:
        return await self._mutate(
            table_id, token, lambda table: table.set_advice_enabled(enabled)
        )

    async def next_hand(self, table_id: str, token: str | None) -> DemoTable:
        table = await self._mutate(table_id, token, lambda value: value.start_hand())
        await self.drive_llm(table_id)
        table, _ = self._load(table_id)
        return table

    async def finish(self, table_id: str, token: str | None) -> DemoTable:
        return await self._mutate(table_id, token, lambda table: table.finish_table())

    async def request_advice(self, table_id: str, token: str | None) -> DemoTable:
        async with self._locks[table_id]:
            table, stored_hash = self._load(table_id)
            self._require_owner(stored_hash, token)
            if not table.advice_enabled or table.controller != "human":
                raise DemoConflictError("advice_not_enabled")
            if not table.snapshot()["canAct"]:
                raise DemoConflictError("not_a_hero_decision")
            captured = (table.version, table.controller_epoch, table.hand.hand_index)
            mode = table.config.provider_mode
            if not self._budget_available(table):
                table.pause_for_provider_failure("live_call_budget_exhausted")
                self.store.save(table, stored_hash)
                return table
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(decide, table), timeout=LLM_TIMEOUT_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary must hand control back
            async with self._locks[table_id]:
                current, stored_hash = self._load(table_id)
                current.pause_for_provider_failure(self._failure_reason(exc, mode, "advice"))
                self.store.save(current, stored_hash)
                return current
        async with self._locks[table_id]:
            current, stored_hash = self._load(table_id)
            current_key = (
                current.version,
                current.controller_epoch,
                current.hand.hand_index if current.hand else -1,
            )
            if current_key != captured or current.controller != "human":
                current.record_stale_llm_response("advice")
            else:
                current.record_provider_call(result.response, purpose="advice")
                current.record_advice(result.advice)
            self.store.save(current, stored_hash)
            return current

    def _budget_available(self, table: DemoTable) -> bool:
        return not (
            table.config.provider_mode == "live_aliyun"
            and int(table.provider_usage["live_calls"]) >= self.live_call_limit
        )

    @staticmethod
    def _failure_reason(exc: BaseException, mode: str, purpose: str) -> str:
        kind = "timeout" if isinstance(exc, TimeoutError) else "provider_failure"
        return f"{purpose}_{kind}_{mode}: {str(exc)[:180]}"

    async def drive_llm(self, table_id: str) -> None:
        async with self._driver_locks[table_id]:
            while True:
                async with self._locks[table_id]:
                    table, stored_hash = self._load(table_id)
                    if table.hand is None:
                        return
                    if table.hand.complete:
                        if not table.completed_hands or (
                            table.completed_hands[-1].get("controller") != "llm_closed_loop"
                        ):
                            return
                        already_reflected = any(
                            int(item.get("handIndex", -1)) == table.hand.hand_index
                            for item in table.reflection_memory
                        )
                        if already_reflected:
                            return
                        purpose = "reflection"
                    elif (
                        table.phase == "waiting_llm"
                        and table.controller_for(table.hand.actor) == "llm_closed_loop"
                    ):
                        purpose = "action"
                    else:
                        return
                    if not self._budget_available(table):
                        table.pause_for_provider_failure("live_call_budget_exhausted")
                        self.store.save(table, stored_hash)
                        return
                    actor = table.hero_seat if purpose == "reflection" else table.hand.actor
                    captured = (
                        table.version,
                        table.controller_epoch,
                        table.hand.hand_index,
                        table.hand.actor,
                    )
                    mode = table.config.provider_mode
                try:
                    if purpose == "action":
                        result = await asyncio.wait_for(
                            asyncio.to_thread(decide, table, actor),
                            timeout=LLM_TIMEOUT_SECONDS,
                        )
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(reflect_and_patch, table),
                            timeout=LLM_TIMEOUT_SECONDS,
                        )
                except Exception as exc:  # noqa: BLE001 - provider boundary must hand control back
                    async with self._locks[table_id]:
                        current, stored_hash = self._load(table_id)
                        current.pause_for_provider_failure(
                            self._failure_reason(exc, mode, purpose)
                        )
                        self.store.save(current, stored_hash)
                    return
                async with self._locks[table_id]:
                    current, stored_hash = self._load(table_id)
                    current_key = (
                        current.version,
                        current.controller_epoch,
                        current.hand.hand_index if current.hand else -1,
                        current.hand.actor if current.hand else -1,
                    )
                    if (
                        current_key != captured
                        or current.controller_for(actor) != "llm_closed_loop"
                    ):
                        current.record_stale_llm_response(purpose)
                        self.store.save(current, stored_hash)
                        return
                    try:
                        current.record_provider_call(result.response, purpose=purpose)
                        if purpose == "action":
                            current.record_advice(result.advice, actor=actor)
                            current.apply_action(actor, result.action, result.raise_scale)
                        else:
                            current.record_reflection(result.reflection)
                            current.apply_strategy_patch(result.patch)
                    except ValueError as exc:
                        current.pause_for_provider_failure(
                            f"{purpose}_validation_failure: {str(exc)[:180]}"
                        )
                        self.store.save(current, stored_hash)
                        return
                    self.store.save(current, stored_hash)
