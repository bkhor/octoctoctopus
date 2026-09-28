import pytest

from agent import cli, db
from agent.embedding_client import StubEmbeddingClient
from agent.llm_client import StubLLMClient
from agent.models import ApplicationState, FieldSpec


def _setup():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    return conn


def test_cmd_submit_creates_queued_application(capsys):
    conn = _setup()

    app_id = cli.cmd_submit(conn, "https://example.com/job")

    app = db.get_application(conn, app_id)
    assert app["url"] == "https://example.com/job"
    assert app["state"] == "queued"
    captured = capsys.readouterr()
    assert str(app_id) in captured.out


def test_cmd_status_lists_all_applications(capsys):
    conn = _setup()
    db.create_application(conn, "https://example.com/a")
    db.create_application(conn, "https://example.com/b")

    cli.cmd_status(conn)

    captured = capsys.readouterr()
    assert "https://example.com/a" in captured.out
    assert "https://example.com/b" in captured.out


def test_cmd_review_prints_fields_and_cover_letter(capsys):
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    field_id = db.add_field(conn, app_id, "Full name", "text")
    db.update_field(conn, field_id, resolved_value="Jane Doe", needs_input=False, resolved_from="profile")
    pending_field_id = db.add_field(conn, app_id, "Why this company?", "textarea")
    db.update_field(conn, pending_field_id, needs_input=True)
    conn.execute("UPDATE applications SET cover_letter_draft = ? WHERE id = ?", ("Dear team...", app_id))
    conn.commit()

    cli.cmd_review(conn, app_id)

    captured = capsys.readouterr()
    assert "Full name" in captured.out
    assert "Jane Doe" in captured.out
    assert "Dear team..." in captured.out
    assert "Why this company?: None (needs input)" in captured.out


def test_cmd_review_shows_options_for_pending_select_field(capsys):
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    field_id = db.add_field(conn, app_id, "Preferred Location", "select", options=["Remote", "New York"])
    db.update_field(conn, field_id, needs_input=True)

    cli.cmd_review(conn, app_id)

    captured = capsys.readouterr()
    assert "1. Remote" in captured.out
    assert "2. New York" in captured.out


class StubFetcher:
    def __init__(self, fields):
        self._fields = fields

    def fetch_fields(self, url):
        return self._fields


class StubFiller:
    def __init__(self):
        self.filled = None
        self.submitted = False

    def fill(self, url, field_values):
        self.filled = field_values

    def submit(self, url, field_values):
        self.submitted = True


def test_cmd_run_processes_queued_application_to_ready_for_review():
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    fields = [FieldSpec(label="Full name", field_type="text")]
    fetcher = StubFetcher(fields)
    filler = StubFiller()
    llm = StubLLMClient()
    llm.register("classify_fields", {
        "fields": [{"index": 0, "is_formal": True}]
    })
    llm.register("cover letter", {"cover_letter": "Dear hiring team, ..."})
    embedder = StubEmbeddingClient()
    profile = {"full_name": "Jane Doe"}

    cli.cmd_run(conn, llm, embedder, fetcher, filler, profile)

    app = db.get_application(conn, app_id)
    assert app["state"] == ApplicationState.READY_FOR_REVIEW.value


def test_cmd_run_skips_application_that_raises_but_continues_the_batch(capsys):
    conn = _setup()
    db.create_application(conn, "https://example.com/broken")
    healthy_app_id = db.create_application(conn, "https://example.com/job")

    class RaisingFetcher:
        def fetch_fields(self, url):
            if url == "https://example.com/broken":
                raise RuntimeError("boom")
            return [FieldSpec(label="Full name", field_type="text")]

    filler = StubFiller()
    llm = StubLLMClient()
    llm.register("classify_fields", {
        "fields": [{"index": 0, "is_formal": True}]
    })
    llm.register("cover letter", {"cover_letter": "Dear hiring team, ..."})
    embedder = StubEmbeddingClient()
    profile = {"full_name": "Jane Doe"}

    failures = cli.cmd_run(conn, llm, embedder, RaisingFetcher(), filler, profile)

    captured = capsys.readouterr()
    assert "boom" in captured.err
    assert "RuntimeError" in captured.err
    assert failures == 1
    healthy_app = db.get_application(conn, healthy_app_id)
    assert healthy_app["state"] == ApplicationState.READY_FOR_REVIEW.value


def test_cmd_run_stops_at_awaiting_hitl():
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    fields = [FieldSpec(label="Why this company?", field_type="textarea")]
    fetcher = StubFetcher(fields)
    filler = StubFiller()
    llm = StubLLMClient()
    llm.register("classify_fields", {
        "fields": [{"index": 0, "is_formal": False}]
    })
    llm.register("Why this company?", {"category": "motivation"})
    embedder = StubEmbeddingClient()

    failures = cli.cmd_run(conn, llm, embedder, fetcher, filler, {})

    assert failures == 0
    app = db.get_application(conn, app_id)
    assert app["state"] == ApplicationState.AWAITING_HITL.value


def test_non_terminal_and_stopping_states_partition_application_state():
    all_values = {state.value for state in ApplicationState}
    non_terminal = set(cli.NON_TERMINAL_STATES)
    stopping = set(cli.STOPPING_STATES)

    assert non_terminal | stopping == all_values
    assert non_terminal & stopping == set()


def test_cmd_hitl_resolves_field_and_advances_state():
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    db.update_application_state(conn, app_id, ApplicationState.AWAITING_HITL.value)
    field_id = db.add_field(conn, app_id, "Why this company?", "textarea")
    db.update_field(conn, field_id, needs_input=True)
    llm = StubLLMClient()
    llm.register("I value mission-driven teams", {"canonical_answer": "I value mission-driven teams"})
    embedder = StubEmbeddingClient()

    cli.cmd_hitl(conn, llm, embedder, field_id, "motivation", "I value mission-driven teams")

    field_row = db.list_fields(conn, app_id)[0]
    assert field_row["resolved_value"] == "I value mission-driven teams"
    assert field_row["needs_input"] == 0
    assert field_row["resolved_from"] == "hitl"


def test_cmd_approve_submits_and_marks_done():
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    db.update_application_state(conn, app_id, ApplicationState.READY_FOR_REVIEW.value)
    filler = StubFiller()

    state = cli.cmd_approve(conn, app_id, filler)

    assert state == ApplicationState.DONE.value
    assert filler.submitted is True


def test_main_submit_and_status_end_to_end(tmp_path, capsys):
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"db_path: {db_path}\n")

    cli.main(["--config", str(config_path), "submit", "https://example.com/job"])
    capsys.readouterr()
    cli.main(["--config", str(config_path), "status"])

    captured = capsys.readouterr()
    assert "https://example.com/job" in captured.out

    cli.main(["--config", str(config_path), "review", "1"])
    review_captured = capsys.readouterr()
    assert "url: https://example.com/job" in review_captured.out


def test_main_raises_on_explicit_missing_config():
    with pytest.raises(FileNotFoundError):
        cli.main(["--config", "/nonexistent/path/config.yaml", "status"])


def test_cmd_remove_deletes_application():
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")

    cli.cmd_remove(conn, app_id)

    try:
        db.get_application(conn, app_id)
        assert False
    except KeyError:
        pass


def test_cmd_rerun_resets_application_and_clears_fields():
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    db.update_application_state(conn, app_id, ApplicationState.READY_FOR_REVIEW.value)
    db.add_field(conn, app_id, "Full name", "text")

    cli.cmd_rerun(conn, app_id)

    app = db.get_application(conn, app_id)
    assert app["state"] == ApplicationState.QUEUED.value
    assert db.list_fields(conn, app_id) == []


def test_cmd_hitl_shell_answers_and_skips_fields(capsys):
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    db.update_application_state(conn, app_id, ApplicationState.AWAITING_HITL.value)
    answered_field_id = db.add_field(conn, app_id, "Why this company?", "textarea")
    db.update_field(conn, answered_field_id, needs_input=True)
    skipped_field_id = db.add_field(conn, app_id, "Favorite color?", "text")
    db.update_field(conn, skipped_field_id, needs_input=True)
    llm = StubLLMClient()
    llm.register("Why this company?", {"category": "motivation"})
    llm.register("I value mission-driven teams", {"canonical_answer": "I value mission-driven teams"})
    embedder = StubEmbeddingClient()
    answers = iter(["I value mission-driven teams", ""])
    input_fn = lambda prompt: next(answers)

    cli.cmd_hitl_shell(conn, llm, embedder, app_id, input_fn=input_fn)

    captured = capsys.readouterr()
    assert "1 answered, 1 skipped, 0 ignored" in captured.out
    answered_row = db.list_fields(conn, app_id)[0]
    assert answered_row["resolved_value"] == "I value mission-driven teams"
    assert answered_row["needs_input"] == 0
    skipped_row = db.list_fields(conn, app_id)[1]
    assert skipped_row["needs_input"] == 1


def test_cmd_hitl_shell_select_field_shows_numbered_options(capsys):
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    field_id = db.add_field(conn, app_id, "Preferred Location", "select", options=["Remote", "New York"])
    db.update_field(conn, field_id, needs_input=True)
    llm = StubLLMClient()
    llm.register("Preferred Location", {"category": "logistics"})
    llm.register("New York", {"canonical_answer": "New York"})
    embedder = StubEmbeddingClient()

    cli.cmd_hitl_shell(conn, llm, embedder, app_id, input_fn=lambda prompt: "2")

    captured = capsys.readouterr()
    assert "1. Remote" in captured.out
    assert "2. New York" in captured.out
    field_row = db.list_fields(conn, app_id)[0]
    assert field_row["resolved_value"] == "New York"
    assert field_row["needs_input"] == 0


def test_cmd_hitl_shell_select_field_invalid_choice_is_skipped(capsys):
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    field_id = db.add_field(conn, app_id, "Preferred Location", "select", options=["Remote", "New York"])
    db.update_field(conn, field_id, needs_input=True)
    llm = StubLLMClient()
    embedder = StubEmbeddingClient()

    cli.cmd_hitl_shell(conn, llm, embedder, app_id, input_fn=lambda prompt: "99")

    captured = capsys.readouterr()
    assert "invalid choice" in captured.out
    assert "0 answered, 1 skipped, 0 ignored" in captured.out
    field_row = db.list_fields(conn, app_id)[0]
    assert field_row["needs_input"] == 1


def test_cmd_hitl_shell_radio_field_shows_numbered_options(capsys):
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    field_id = db.add_field(conn, app_id, "Require visa sponsorship?", "radio", options=["Yes", "No"])
    db.update_field(conn, field_id, needs_input=True)
    llm = StubLLMClient()
    llm.register("Require visa sponsorship?", {"category": "eligibility"})
    llm.register("No", {"canonical_answer": "No"})
    embedder = StubEmbeddingClient()

    cli.cmd_hitl_shell(conn, llm, embedder, app_id, input_fn=lambda prompt: "2")

    captured = capsys.readouterr()
    assert "1. Yes" in captured.out
    assert "2. No" in captured.out
    field_row = db.list_fields(conn, app_id)[0]
    assert field_row["resolved_value"] == "No"
    assert field_row["needs_input"] == 0


def test_cmd_review_shows_options_for_pending_radio_field(capsys):
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    field_id = db.add_field(conn, app_id, "Require visa sponsorship?", "radio", options=["Yes", "No"])
    db.update_field(conn, field_id, needs_input=True)

    cli.cmd_review(conn, app_id)

    captured = capsys.readouterr()
    assert "1. Yes" in captured.out
    assert "2. No" in captured.out


def test_cmd_hitl_shell_permanently_ignores_field_on_keyword():
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    field_id = db.add_field(conn, app_id, "field_6", "text")
    db.update_field(conn, field_id, needs_input=True)
    llm = StubLLMClient()
    embedder = StubEmbeddingClient()

    cli.cmd_hitl_shell(conn, llm, embedder, app_id, input_fn=lambda prompt: "skip")

    field_row = db.list_fields(conn, app_id)[0]
    assert field_row["needs_input"] == 0
    assert field_row["resolved_value"] == ""
    assert field_row["resolved_from"] == "ignored"


def test_cmd_hitl_shell_reports_pending_file_fields_without_prompting(capsys):
    conn = _setup()
    app_id = db.create_application(conn, "https://example.com/job")
    field_id = db.add_field(conn, app_id, "Resume", "file")
    db.update_field(conn, field_id, needs_input=True)
    llm = StubLLMClient()
    embedder = StubEmbeddingClient()

    def _unexpected_input(prompt):
        raise AssertionError("should not prompt for file fields")

    cli.cmd_hitl_shell(conn, llm, embedder, app_id, input_fn=_unexpected_input)

    captured = capsys.readouterr()
    assert "Resume" in captured.out
    assert "0 answered, 0 skipped" in captured.out


def test_main_remove_and_rerun_end_to_end(tmp_path, capsys):
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"db_path: {db_path}\n")

    cli.main(["--config", str(config_path), "submit", "https://example.com/job"])
    capsys.readouterr()

    cli.main(["--config", str(config_path), "rerun", "1"])
    cli.main(["--config", str(config_path), "status"])
    status_captured = capsys.readouterr()
    assert "queued" in status_captured.out

    cli.main(["--config", str(config_path), "remove", "1"])
    cli.main(["--config", str(config_path), "status"])
    final_captured = capsys.readouterr()
    assert "https://example.com/job" not in final_captured.out
