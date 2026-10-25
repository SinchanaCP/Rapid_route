"""
database.py

SQLite persistence for RapidRoute dispatches. Uses only the standard
library sqlite3 module plus parameterized queries throughout (no string-
formatted SQL, to avoid injection and match Phase 36's requirement).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[1] / "ambulance.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS dispatches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    origin_source TEXT NOT NULL,           -- 'gps' or 'manual'
    standard_distance_m REAL,
    optimized_distance_m REAL,
    standard_time_min REAL,
    optimized_time_min REAL,
    time_saved_min REAL,
    congestion_level TEXT,
    predicted_traffic REAL,                -- predicted avg speed km/h on the binding road
    reroute_count INTEGER NOT NULL DEFAULT 0
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(SCHEMA)


def save_dispatch(
    origin: str,
    destination: str,
    origin_source: str,
    standard_distance_m: Optional[float],
    optimized_distance_m: Optional[float],
    standard_time_min: Optional[float],
    optimized_time_min: Optional[float],
    congestion_level: Optional[str],
    predicted_traffic: Optional[float],
    reroute_count: int = 0,
) -> int:
    time_saved = None
    if standard_time_min is not None and optimized_time_min is not None:
        time_saved = round(standard_time_min - optimized_time_min, 2)

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO dispatches (
                timestamp, origin, destination, origin_source,
                standard_distance_m, optimized_distance_m,
                standard_time_min, optimized_time_min, time_saved_min,
                congestion_level, predicted_traffic, reroute_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                origin, destination, origin_source,
                standard_distance_m, optimized_distance_m,
                standard_time_min, optimized_time_min, time_saved,
                congestion_level, predicted_traffic, reroute_count,
            ),
        )
        return cur.lastrowid


def increment_reroute_count(dispatch_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE dispatches SET reroute_count = reroute_count + 1 WHERE id = ?",
            (dispatch_id,),
        )


def get_all_dispatches(limit: int = 100) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM dispatches ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_summary_stats() -> dict:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_dispatches,
                AVG(time_saved_min) AS avg_time_saved,
                SUM(time_saved_min) AS total_time_saved,
                SUM(reroute_count) AS total_reroutes
            FROM dispatches
            """
        ).fetchone()

        congestion_row = conn.execute(
            """
            SELECT congestion_level, COUNT(*) AS cnt
            FROM dispatches
            WHERE congestion_level IS NOT NULL
            GROUP BY congestion_level
            """
        ).fetchall()

    return {
        "total_dispatches": row["total_dispatches"] or 0,
        "avg_time_saved_min": round(row["avg_time_saved"], 2) if row["avg_time_saved"] is not None else None,
        "total_time_saved_min": round(row["total_time_saved"], 2) if row["total_time_saved"] is not None else None,
        "total_reroutes": row["total_reroutes"] or 0,
        "congestion_breakdown": {r["congestion_level"]: r["cnt"] for r in congestion_row},
    }
