"""Load traces into SQLite and query them back.

SQLite rather than a dataframe because the UI needs filtering, sorting and counts
over a few thousand rows and SQL already does that well. It is a single file, so
the whole thing still runs with `python -m app.main` and no services to start.

The jsonl stays the source of truth. Ingest is idempotent: re-running it drops and
rebuilds the table, so there is never a half-migrated state to reason about.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from app.schema import Trace, from_dict

DB_PATH = Path("data/traces.db")

SCHEMA = """
CREATE TABLE traces (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    ts              TEXT NOT NULL,
    model           TEXT NOT NULL,
    system_prompt   TEXT,
    messages        TEXT NOT NULL,     -- json
    response        TEXT,
    tool_calls      TEXT NOT NULL,     -- json
    finish_reason   TEXT NOT NULL,
    status          INTEGER NOT NULL,
    prompt_tokens   INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens    INTEGER NOT NULL,
    cost_usd        REAL NOT NULL,
    latency_ms      INTEGER NOT NULL,
    feedback        TEXT,
    retry_of        TEXT,
    continuation    TEXT,
    truncated       INTEGER NOT NULL,
    failed          INTEGER NOT NULL,
    tool_error      INTEGER NOT NULL,
    turn_count      INTEGER NOT NULL
);
CREATE INDEX idx_session ON traces(session_id);
CREATE INDEX idx_model   ON traces(model);
CREATE INDEX idx_ts      ON traces(ts);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def read_jsonl(path: Path) -> list[Trace]:
    traces = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            traces.append(from_dict(json.loads(line)))
        except Exception as exc:
            # A corrupt line should not sink the whole ingest. Say which one.
            print(f"  skipped line {lineno}: {exc}")
    return traces


def ingest(conn: sqlite3.Connection, traces: Iterable[Trace]) -> int:
    conn.executescript("DROP TABLE IF EXISTS traces;")
    conn.executescript(SCHEMA)
    rows = []
    for t in traces:
        rows.append((
            t.id, t.session_id, t.ts, t.model, t.system_prompt,
            json.dumps(t.messages), t.response,
            json.dumps([tc.__dict__ for tc in t.tool_calls]),
            t.finish_reason, t.status, t.prompt_tokens, t.completion_tokens,
            t.total_tokens, t.cost_usd, t.latency_ms, t.feedback, t.retry_of,
            t.continuation, int(t.truncated), int(t.failed), int(t.tool_error),
            t.turn_count,
        ))
    conn.executemany(
        "INSERT INTO traces VALUES (" + ",".join("?" * 22) + ")", rows)
    conn.commit()
    return len(rows)


# Filters the UI exposes. Kept as a dict so adding one means adding a line here
# and a control in the template, not touching the query builder.
FILTERS: dict[str, str] = {
    "model":       "model = :model",
    "feedback":    "feedback = :feedback",
    "finish":      "finish_reason = :finish",
    "truncated":   "truncated = 1",
    "errors":      "failed = 1",
    "tool_errors": "tool_error = 1",
    "retries":     "retry_of IS NOT NULL",
    "min_cost":    "cost_usd >= :min_cost",
    "min_latency": "latency_ms >= :min_latency",
    "session":     "session_id = :session",
}
BOOLEAN_FILTERS = {"truncated", "errors", "tool_errors", "retries"}

SORTS = {
    "ts": "ts DESC", "cost": "cost_usd DESC", "latency": "latency_ms DESC",
    "tokens": "total_tokens DESC", "turns": "turn_count DESC",
}


def build_where(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    clauses, bound = [], {}
    for key, sql in FILTERS.items():
        val = params.get(key)
        if val in (None, "", "any"):
            continue
        if key in BOOLEAN_FILTERS:
            if str(val).lower() in ("1", "true", "on", "yes"):
                clauses.append(sql)
        else:
            clauses.append(sql)
            bound[key] = float(val) if key.startswith("min_") else val
    q = params.get("q")
    if q:
        clauses.append("(response LIKE :q OR messages LIKE :q OR system_prompt LIKE :q)")
        bound["q"] = f"%{q}%"
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", bound


def query(conn: sqlite3.Connection, params: dict[str, Any],
          limit: int = 100, offset: int = 0) -> tuple[list[sqlite3.Row], int]:
    where, bound = build_where(params)
    order = SORTS.get(params.get("sort") or "ts", SORTS["ts"])
    total = conn.execute(f"SELECT COUNT(*) FROM traces{where}", bound).fetchone()[0]
    # limit and offset are cast to int rather than bound, because mixing named
    # and positional parameters in one statement is not worth the confusion.
    rows = conn.execute(
        f"SELECT * FROM traces{where} ORDER BY {order} "
        f"LIMIT {int(limit)} OFFSET {int(offset)}", bound).fetchall()
    return rows, total


def summary(conn: sqlite3.Connection, params: dict[str, Any]) -> dict[str, Any]:
    where, bound = build_where(params)
    row = conn.execute(f"""
        SELECT COUNT(*) n,
               COALESCE(SUM(cost_usd), 0) cost,
               COALESCE(SUM(total_tokens), 0) tokens,
               COALESCE(AVG(latency_ms), 0) latency,
               COALESCE(SUM(failed), 0) failed,
               COALESCE(SUM(truncated), 0) truncated,
               COALESCE(SUM(tool_error), 0) tool_errors,
               COUNT(DISTINCT session_id) sessions
        FROM traces{where}""", bound).fetchone()
    return dict(row)


def models(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT model FROM traces ORDER BY model")]


def by_model(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("""
        SELECT model,
               COUNT(*) n,
               SUM(cost_usd) cost,
               AVG(latency_ms) latency,
               SUM(truncated) truncated,
               SUM(failed) failed,
               SUM(CASE WHEN feedback='strong' THEN 1 ELSE 0 END) strong,
               SUM(CASE WHEN feedback='weak' THEN 1 ELSE 0 END) weak
        FROM traces GROUP BY model ORDER BY cost DESC""")]


def get(conn: sqlite3.Connection, trace_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()


def session_traces(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM traces WHERE session_id = ? ORDER BY ts", (session_id,)).fetchall()


def all_traces(conn: sqlite3.Connection) -> list[Trace]:
    out = []
    for r in conn.execute("SELECT * FROM traces ORDER BY ts"):
        d = dict(r)
        d["messages"] = json.loads(d["messages"])
        d["tool_calls"] = json.loads(d["tool_calls"])
        out.append(from_dict(d))
    return out
