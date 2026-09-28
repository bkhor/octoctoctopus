import sqlite3
from agent import db


def test_init_schema_creates_all_tables():
    conn = db.connect(":memory:")
    db.init_schema(conn)

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    assert {"applications", "application_fields", "memory"} <= tables


def test_create_and_get_application():
    conn = db.connect(":memory:")
    db.init_schema(conn)

    app_id = db.create_application(conn, "https://boards.greenhouse.io/acme/jobs/1")

    row = db.get_application(conn, app_id)
    assert row["url"] == "https://boards.greenhouse.io/acme/jobs/1"
    assert row["state"] == "queued"
    assert row["created_at"] == row["updated_at"]


def test_update_application_state_bumps_updated_at():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    app_id = db.create_application(conn, "https://example.com/job")
    before = db.get_application(conn, app_id)["updated_at"]

    db.update_application_state(conn, app_id, "fetching")

    row = db.get_application(conn, app_id)
    assert row["state"] == "fetching"
    assert row["updated_at"] >= before


def test_list_applications_by_state():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    a1 = db.create_application(conn, "https://example.com/1")
    a2 = db.create_application(conn, "https://example.com/2")
    db.update_application_state(conn, a2, "fetching")

    queued = db.list_applications_by_state(conn, "queued")

    assert [row["id"] for row in queued] == [a1]


def test_add_and_list_fields():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    app_id = db.create_application(conn, "https://example.com/job")

    field_id = db.add_field(conn, app_id, "Full name", "text", category="formal")
    db.update_field(conn, field_id, resolved_value="Jane Doe", resolved_from="profile", needs_input=False)

    fields = db.list_fields(conn, app_id)
    assert len(fields) == 1
    assert fields[0]["label"] == "Full name"
    assert fields[0]["resolved_value"] == "Jane Doe"
    assert fields[0]["needs_input"] == 0


def test_delete_application_removes_application_and_fields():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    app_id = db.create_application(conn, "https://example.com/job")
    db.add_field(conn, app_id, "Full name", "text")

    db.delete_application(conn, app_id)

    try:
        db.get_application(conn, app_id)
        assert False
    except KeyError:
        pass
    assert db.list_fields(conn, app_id) == []


def test_delete_fields_for_application_clears_only_that_applications_fields():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    app_id = db.create_application(conn, "https://example.com/job")
    other_app_id = db.create_application(conn, "https://example.com/other")
    db.add_field(conn, app_id, "Full name", "text")
    db.add_field(conn, other_app_id, "Email", "text")

    db.delete_fields_for_application(conn, app_id)

    assert db.list_fields(conn, app_id) == []
    assert len(db.list_fields(conn, other_app_id)) == 1


def test_reset_application_returns_to_queued_and_clears_cover_letter():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    app_id = db.create_application(conn, "https://example.com/job")
    db.update_application_state(conn, app_id, "ready_for_review")
    conn.execute("UPDATE applications SET cover_letter_draft = ? WHERE id = ?", ("Dear team...", app_id))
    conn.commit()

    db.reset_application(conn, app_id)

    row = db.get_application(conn, app_id)
    assert row["state"] == "queued"
    assert row["cover_letter_draft"] is None


def test_add_and_query_memory_by_category():
    conn = db.connect(":memory:")
    db.init_schema(conn)

    db.add_memory_entry(conn, "Why this company?", "I care about X", b"\x00\x01", "motivation")
    db.add_memory_entry(conn, "Biggest failure?", "I once shipped a bug", b"\x02\x03", "failure-story")

    motivation_rows = db.list_memory_by_category(conn, "motivation")
    assert len(motivation_rows) == 1
    assert motivation_rows[0]["canonical_answer"] == "I care about X"
    assert motivation_rows[0]["embedding"] == b"\x00\x01"
