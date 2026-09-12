from agent import db, orchestrator
from agent.embedding_client import StubEmbeddingClient
from agent.llm_client import StubLLMClient
from agent.models import ApplicationState, FieldSpec


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

    def submit(self, url):
        self.submitted = True


def test_full_application_lifecycle_queued_to_done():
    conn = db.connect(":memory:")
    db.init_schema(conn)
    app_id = db.create_application(conn, "https://boards.greenhouse.io/acme/jobs/42")

    fields = [
        FieldSpec(label="Full name", field_type="text"),
        FieldSpec(label="Why do you want to work here?", field_type="textarea"),
    ]
    fetcher = StubFetcher(fields)
    filler = StubFiller()
    profile = {"full_name": "Jane Doe"}

    llm = StubLLMClient()
    llm.register("classify_fields", {
        "fields": [
            {"label": "Full name", "category": "formal", "profile_key": "full_name"},
            {"label": "Why do you want to work here?", "category": "motivation", "profile_key": ""},
        ]
    })
    llm.register("Why do you want to work here?", {"category": "motivation"})
    llm.register("mission-driven work excites me", {"canonical_answer": "I'm drawn to mission-driven teams"})
    llm.register("cover letter", {"cover_letter": "Dear Acme team, I'm excited to apply..."})
    embedder = StubEmbeddingClient()

    state = None
    for _ in range(10):
        state = orchestrator.step(
            conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile
        )
        if state in (ApplicationState.AWAITING_HITL.value, ApplicationState.READY_FOR_REVIEW.value):
            break

    assert state == ApplicationState.AWAITING_HITL.value

    hitl_field = next(
        row for row in db.list_fields(conn, app_id)
        if row["label"] == "Why do you want to work here?"
    )
    orchestrator.resume_after_hitl(
        conn, app_id, llm=llm, embedder=embedder,
        field_id=hitl_field["id"], category="motivation",
        answer_text="mission-driven work excites me",
    )

    for _ in range(10):
        state = orchestrator.step(
            conn, app_id, llm=llm, embedder=embedder, fetcher=fetcher, filler=filler, profile=profile
        )
        if state == ApplicationState.READY_FOR_REVIEW.value:
            break

    assert state == ApplicationState.READY_FOR_REVIEW.value
    assert filler.filled == {
        "Full name": "Jane Doe",
        "Why do you want to work here?": "mission-driven work excites me",
    }

    final_state = orchestrator.approve(conn, app_id, filler=filler)

    assert final_state == ApplicationState.DONE.value
    assert filler.submitted is True

    memory_rows = db.list_memory_by_category(conn, "motivation")
    assert len(memory_rows) == 1
    assert memory_rows[0]["canonical_answer"] == "I'm drawn to mission-driven teams"
