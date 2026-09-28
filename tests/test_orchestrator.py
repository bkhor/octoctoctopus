from agent import db, memory, orchestrator
from agent.embedding_client import StubEmbeddingClient
from agent.llm_client import StubLLMClient
from agent.models import ApplicationState, FieldSpec, FormUnparseable


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
        self.last_submit_field_values = None

    def fill(self, url: str, field_values: dict[str, str]) -> None:
        self.filled = field_values

    def submit(self, url: str, field_values: dict[str, str]) -> None:
        self.submitted = True
        self.submit_call_count += 1
        self.last_submit_field_values = field_values


class RaisingFetcher:
    def fetch_fields(self, url: str) -> list[FieldSpec]:
        raise FormUnparseable("could not resolve enough labels")


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
        "fields": [{"index": 0, "is_formal": True}]
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
        "fields": [{"index": 0, "is_formal": False}]
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
        "fields": [{"index": 0, "is_formal": True}]
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
    field_id = db.add_field(conn, app_id, "Full name", "text")
    db.update_field(conn, field_id, resolved_value="Jane Doe", needs_input=False, resolved_from="profile")
    filler = StubFiller()

    state = orchestrator.approve(conn, app_id, filler=filler)

    assert state == ApplicationState.DONE.value
    assert filler.submitted is True
    assert filler.last_submit_field_values == {"Full name": "Jane Doe"}


def test_approve_is_safe_to_call_twice():
    conn, app_id = _setup()
    db.update_application_state(conn, app_id, ApplicationState.READY_FOR_REVIEW.value)
    filler = StubFiller()

    orchestrator.approve(conn, app_id, filler=filler)
    second_state = orchestrator.approve(conn, app_id, filler=filler)

    assert second_state == ApplicationState.DONE.value
    assert filler.submit_call_count == 1


def test_fetching_with_unparseable_form_goes_to_manual_fallback():
    conn, app_id = _setup()
    fetcher = RaisingFetcher()
    llm = StubLLMClient()
    embedder = StubEmbeddingClient()

    orchestrator.step(
        conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=StubFiller(), profile={}
    )
    state = orchestrator.step(
        conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=StubFiller(), profile={}
    )

    assert state == ApplicationState.MANUAL_FALLBACK.value
    assert db.get_application(conn, app_id)["state"] == "manual_fallback"


def test_file_field_auto_resolves_from_profile_regardless_of_category():
    conn, app_id = _setup()
    fields = [FieldSpec(label="Resume", field_type="file")]
    fetcher = StubFetcher(fields)
    filler = StubFiller()
    llm = StubLLMClient()
    llm.register("classify_fields", {
        "fields": [{"index": 0, "is_formal": False}]
    })
    llm.register("cover letter", {"cover_letter": "Dear hiring team, ..."})
    embedder = StubEmbeddingClient()
    profile = {"resume": "/home/user/resume.pdf"}

    state = None
    for _ in range(10):
        state = orchestrator.step(
            conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile
        )
        if state in (ApplicationState.AWAITING_HITL.value, ApplicationState.READY_FOR_REVIEW.value):
            break

    field_row = db.list_fields(conn, app_id)[0]
    assert field_row["resolved_value"] == "/home/user/resume.pdf"
    assert field_row["resolved_from"] == "static_attachment"
    assert field_row["needs_input"] == 0


def test_file_field_without_profile_path_needs_input():
    conn, app_id = _setup()
    fields = [FieldSpec(label="Resume", field_type="file")]
    fetcher = StubFetcher(fields)
    filler = StubFiller()
    llm = StubLLMClient()
    llm.register("classify_fields", {
        "fields": [{"index": 0, "is_formal": False}]
    })
    embedder = StubEmbeddingClient()
    profile = {}

    orchestrator.step(conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile)
    orchestrator.step(conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile)
    orchestrator.step(conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile)
    orchestrator.step(conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile)

    field_row = db.list_fields(conn, app_id)[0]
    assert field_row["needs_input"] == 1


def test_file_field_with_memory_match_is_not_clobbered_by_synthesis():
    conn, app_id = _setup()
    fields = [FieldSpec(label="Resume", field_type="file")]
    fetcher = StubFetcher(fields)
    filler = StubFiller()
    llm = StubLLMClient()
    llm.register("classify_fields", {
        "fields": [{"index": 0, "is_formal": False}]
    })
    llm.register("Resume", {"category": "attachment", "answer": "I have attached my resume"})
    embedder = StubEmbeddingClient()
    vector = embedder.embed("Resume")
    db.add_memory_entry(conn, "Resume", "canonical resume note one", memory._pack(vector), "attachment")
    db.add_memory_entry(conn, "Resume", "canonical resume note two", memory._pack(vector), "attachment")
    profile = {}

    state = None
    for _ in range(10):
        state = orchestrator.step(
            conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile
        )
        if state in (ApplicationState.AWAITING_HITL.value, ApplicationState.READY_FOR_REVIEW.value):
            break

    field_row = db.list_fields(conn, app_id)[0]
    assert field_row["resolved_from"] != "memory"
    assert field_row["field_type"] == "file"
    assert field_row["needs_input"] == 1
    assert state == ApplicationState.AWAITING_HITL.value
