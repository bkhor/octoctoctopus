import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    company TEXT,
    role TEXT,
    state TEXT NOT NULL,
    cover_letter_draft TEXT,
    status TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS application_fields (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id),
    label TEXT NOT NULL,
    field_type TEXT NOT NULL,
    category TEXT,
    resolved_value TEXT,
    needs_input INTEGER NOT NULL DEFAULT 0,
    resolved_from TEXT
);

CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    canonical_answer TEXT NOT NULL,
    embedding BLOB NOT NULL,
    category_tag TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_application(conn: sqlite3.Connection, url: str) -> int:
    now = _now()
    cur = conn.execute(
        "INSERT INTO applications (url, state, created_at, updated_at) "
        "VALUES (?, 'queued', ?, ?)",
        (url, now, now),
    )
    conn.commit()
    return cur.lastrowid


def get_application(conn: sqlite3.Connection, app_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM applications WHERE id = ?", (app_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"no application with id {app_id}")
    return row


def update_application_state(conn: sqlite3.Connection, app_id: int, new_state: str) -> None:
    conn.execute(
        "UPDATE applications SET state = ?, updated_at = ? WHERE id = ?",
        (new_state, _now(), app_id),
    )
    conn.commit()


def list_applications_by_state(conn: sqlite3.Connection, state: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM applications WHERE state = ? ORDER BY created_at", (state,)
    ).fetchall()


def add_field(
    conn: sqlite3.Connection,
    application_id: int,
    label: str,
    field_type: str,
    category: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO application_fields (application_id, label, field_type, category, needs_input) "
        "VALUES (?, ?, ?, ?, 0)",
        (application_id, label, field_type, category),
    )
    conn.commit()
    return cur.lastrowid


def update_field(
    conn: sqlite3.Connection,
    field_id: int,
    *,
    resolved_value: str | None = None,
    needs_input: bool | None = None,
    resolved_from: str | None = None,
) -> None:
    current = conn.execute(
        "SELECT resolved_value, needs_input, resolved_from FROM application_fields WHERE id = ?",
        (field_id,),
    ).fetchone()
    if current is None:
        raise KeyError(f"no field with id {field_id}")

    new_value = current["resolved_value"] if resolved_value is None else resolved_value
    new_needs_input = current["needs_input"] if needs_input is None else int(needs_input)
    new_resolved_from = current["resolved_from"] if resolved_from is None else resolved_from

    conn.execute(
        "UPDATE application_fields SET resolved_value = ?, needs_input = ?, resolved_from = ? WHERE id = ?",
        (new_value, new_needs_input, new_resolved_from, field_id),
    )
    conn.commit()


def list_fields(conn: sqlite3.Connection, application_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM application_fields WHERE application_id = ? ORDER BY id",
        (application_id,),
    ).fetchall()


def add_memory_entry(
    conn: sqlite3.Connection,
    question: str,
    canonical_answer: str,
    embedding: bytes,
    category_tag: str,
) -> int:
    now = _now()
    cur = conn.execute(
        "INSERT INTO memory (question, canonical_answer, embedding, category_tag, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (question, canonical_answer, embedding, category_tag, now, now),
    )
    conn.commit()
    return cur.lastrowid


def list_memory_by_category(conn: sqlite3.Connection, category_tag: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM memory WHERE category_tag = ? ORDER BY created_at", (category_tag,)
    ).fetchall()
