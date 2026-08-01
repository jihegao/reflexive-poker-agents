from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .demo_engine import DemoTable


def owner_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class DemoStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS demo_tables (
                    table_id TEXT PRIMARY KEY,
                    owner_hash TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS demo_tables_owner_updated
                    ON demo_tables(owner_hash, updated_at DESC);
                CREATE TABLE IF NOT EXISTS demo_events (
                    table_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(table_id, seq),
                    FOREIGN KEY(table_id) REFERENCES demo_tables(table_id) ON DELETE CASCADE
                );
                """
            )

    def save(self, table: DemoTable, token_hash: str) -> None:
        events = table.drain_events()
        state_json = json.dumps(table.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO demo_tables(table_id, owner_hash, state_json)
                VALUES (?, ?, ?)
                ON CONFLICT(table_id) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (table.table_id, token_hash, state_json),
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO demo_events(table_id, seq, event_type, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        table.table_id,
                        int(event["seq"]),
                        str(event["type"]),
                        json.dumps(event["payload"], ensure_ascii=False),
                    )
                    for event in events
                ],
            )

    def load(self, table_id: str) -> tuple[DemoTable, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json, owner_hash FROM demo_tables WHERE table_id=?", (table_id,)
            ).fetchone()
        if row is None:
            return None
        return DemoTable.from_dict(json.loads(row["state_json"])), str(row["owner_hash"])

    def active_for_owner(self, token_hash: str) -> DemoTable | None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT state_json FROM demo_tables
                WHERE owner_hash=? ORDER BY updated_at DESC
                """,
                (token_hash,),
            ).fetchall()
        for row in rows:
            table = DemoTable.from_dict(json.loads(row["state_json"]))
            if not table.ended:
                return table
        return None

    def events_after(self, table_id: str, seq: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT seq, event_type, payload_json, created_at
                FROM demo_events WHERE table_id=? AND seq>? ORDER BY seq LIMIT 100
                """,
                (table_id, seq),
            ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "type": str(row["event_type"]),
                "payload": json.loads(row["payload_json"]),
                "createdAt": str(row["created_at"]),
            }
            for row in rows
        ]
