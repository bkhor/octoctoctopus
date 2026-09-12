from agent import db, orchestrator
from agent.embedding_client import StubEmbeddingClient
from agent.llm_client import StubLLMClient
from agent.models import ApplicationState, FieldSpec


class StubFetcher:
    def __init__(self, fields: list[FieldSpec]):
        self._fields = fields

    def fetch_fields(self, url: str) -> list[FieldSpec]:
        return self._fields


class StubFiller:
    def __init__(self):
        self.filled = None
        self.submitted = False
        self.submit_call_count = 0

    def fill(self, url: str, field_values: dict[str, str]) -> None:
        self.filled = field_values

    def submit(self, url: str) -> None:
        self.submitted = True
        self.submit_call_count += 1


def _setup():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    app_id = db.create_application(conn, "https://example.com/job")
    return conn, app_id


def test_step_advances_queued_to_fetching():
    conn, app_id = _setup()
    fields = [FieldSpec(label="Full name", field_type="text")]
    fetcher = StubFetcher(fields)
    llm = StubLLMClient()
    embedder = StubEmbeddingClient()
    profile = {"full_name": "Jane Doe"}

    new_state = orchestrator.step(
        conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=StubFiller(), profile=profile
    )

    assert new_state == ApplicationState.FETCHING.value
    assert db.get_application(conn, app_id)["state"] == "fetching"


def test_full_happy_path_reaches_ready_for_review():
    conn, app_id = _setup()
    fields = [
        FieldSpec(label="Full name", field_type="text"),
    ]
    fetcher = StubFetcher(fields)
    filler = StubFiller()
    llm = StubLLMClient()
    llm.register("classify_fields", {
        "fields": [{"label": "Full name", "category": "formal", "profile_key": "full_name"}]
    })
    llm.register("cover letter", {"cover_letter": "Dear hiring team, ..."})
    embedder = StubEmbeddingClient()
    profile = {"full_name": "Jane Doe"}

    state = None
    for _ in range(10):
        state = orchestrator.step(
            conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile
        )
        if state in (ApplicationState.READY_FOR_REVIEW.value, ApplicationState.AWAITING_HITL.value):
            break

    assert state == ApplicationState.READY_FOR_REVIEW.value
    assert filler.filled == {"Full name": "Jane Doe"}


def test_hitl_gap_pauses_at_awaiting_hitl_then_resumes():
    conn, app_id = _setup()
    fields = [FieldSpec(label="Why this company?", field_type="textarea")]
    fetcher = StubFetcher(fields)
    filler = StubFiller()
    llm = StubLLMClient()
    llm.register("classify_fields", {
        "fields": [{"label": "Why this company?", "category": "motivation", "profile_key": ""}]
    })
    llm.register("Why this company?", {"category": "motivation"})
    llm.register("raw HITL answer", {"canonical_answer": "I value mission-driven teams"})
    llm.register("cover letter", {"cover_letter": "Dear hiring team, ..."})
    embedder = StubEmbeddingClient()
    profile = {}

    state = None
    for _ in range(10):
        state = orchestrator.step(
            conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile
        )
        if state == ApplicationState.AWAITING_HITL.value:
            break

    assert state == ApplicationState.AWAITING_HITL.value
    field_row = db.list_fields(conn, app_id)[0]
    assert field_row["needs_input"] == 1

    orchestrator.resume_after_hitl(
        conn, app_id, llm=llm, embedder=embedder,
        field_id=field_row["id"], category="motivation", answer_text="raw HITL answer",
    )

    for _ in range(10):
        state = orchestrator.step(
            conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile
        )
        if state == ApplicationState.READY_FOR_REVIEW.value:
            break

    assert state == ApplicationState.READY_FOR_REVIEW.value
    resolved_field = db.list_fields(conn, app_id)[0]
    assert resolved_field["needs_input"] == 0
    assert resolved_field["resolved_from"] == "hitl"


def test_step_is_safe_to_call_after_simulated_restart():
    conn, app_id = _setup()
    fields = [FieldSpec(label="Full name", field_type="text")]
    fetcher = StubFetcher(fields)
    filler = StubFiller()
    llm = StubLLMClient()
    llm.register("classify_fields", {
        "fields": [{"label": "Full name", "category": "formal", "profile_key": "full_name"}]
    })
    embedder = StubEmbeddingClient()
    profile = {"full_name": "Jane Doe"}

    orchestrator.step(conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile)
    orchestrator.step(conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile)

    reloaded = db.get_application(conn, app_id)
    assert reloaded["state"] == ApplicationState.CLASSIFYING.value

    orchestrator.step(conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile)
    assert len(db.list_fields(conn, app_id)) == 1


def test_approve_submits_and_marks_done():
    conn, app_id = _setup()
    db.update_application_state(conn, app_id, ApplicationState.READY_FOR_REVIEW.value)
    filler = StubFiller()

    state = orchestrator.approve(conn, app_id, filler=filler)

    assert state == ApplicationState.DONE.value
    assert filler.submitted is True


def test_approve_is_safe_to_call_twice():
    conn, app_id = _setup()
    db.update_application_state(conn, app_id, ApplicationState.READY_FOR_REVIEW.value)
    filler = StubFiller()

    orchestrator.approve(conn, app_id, filler=filler)
    second_state = orchestrator.approve(conn, app_id, filler=filler)

    assert second_state == ApplicationState.DONE.value
    assert filler.submit_call_count == 1
