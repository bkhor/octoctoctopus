from typing import Protocol

from . import db, memory
from .embedding_client import EmbeddingClient
from .llm_client import LLMClient
from .models import ApplicationState, FieldSpec

FORMAL_CATEGORY = "formal"


class PageFetcher(Protocol):
    def fetch_fields(self, url: str) -> list[FieldSpec]:
        ...


class FormFiller(Protocol):
    def fill(self, url: str, field_values: dict[str, str]) -> None:
        ...

    def submit(self, url: str) -> None:
        ...


def step(
    conn,
    app_id: int,
    *,
    llm: LLMClient,
    embedder: EmbeddingClient,
    fetcher: PageFetcher,
    filler: FormFiller,
    profile: dict[str, str],
) -> str:
    app = db.get_application(conn, app_id)
    state = app["state"]

    if state == ApplicationState.QUEUED.value:
        return _to(conn, app_id, ApplicationState.FETCHING)

    if state == ApplicationState.FETCHING.value:
        fields = fetcher.fetch_fields(app["url"])
        for f in fields:
            db.add_field(conn, app_id, f.label, f.field_type)
        return _to(conn, app_id, ApplicationState.CLASSIFYING)

    if state == ApplicationState.CLASSIFYING.value:
        _classify_fields(conn, app_id, llm)
        return _to(conn, app_id, ApplicationState.RESOLVING_FORMAL)

    if state == ApplicationState.RESOLVING_FORMAL.value:
        _resolve_formal_fields(conn, app_id, profile)
        return _to(conn, app_id, ApplicationState.RESOLVING_INFORMAL)

    if state == ApplicationState.RESOLVING_INFORMAL.value:
        gap = _resolve_informal_fields(conn, app_id, llm, embedder)
        if gap:
            return _to(conn, app_id, ApplicationState.AWAITING_HITL)
        return _to(conn, app_id, ApplicationState.DRAFTING_COVER_LETTER)

    if state == ApplicationState.AWAITING_HITL.value:
        return state

    if state == ApplicationState.DRAFTING_COVER_LETTER.value:
        _draft_cover_letter(conn, app_id, llm)
        return _to(conn, app_id, ApplicationState.FILLING)

    if state == ApplicationState.FILLING.value:
        field_values = {
            row["label"]: row["resolved_value"] for row in db.list_fields(conn, app_id)
        }
        filler.fill(app["url"], field_values)
        return _to(conn, app_id, ApplicationState.READY_FOR_REVIEW)

    return state


def resume_after_hitl(
    conn,
    app_id: int,
    *,
    llm: LLMClient,
    embedder: EmbeddingClient,
    field_id: int,
    category: str,
    answer_text: str,
) -> str:
    field_row = conn.execute(
        "SELECT label FROM application_fields WHERE id = ?", (field_id,)
    ).fetchone()
    memory.store_answer(conn, llm, embedder, field_row["label"], answer_text, category)
    db.update_field(conn, field_id, resolved_value=answer_text, needs_input=False, resolved_from="hitl")
    return _to(conn, app_id, ApplicationState.RESOLVING_INFORMAL)


def approve(conn, app_id: int, *, filler: FormFiller) -> str:
    app = db.get_application(conn, app_id)
    if app["state"] != ApplicationState.READY_FOR_REVIEW.value:
        return app["state"]
    filler.submit(app["url"])
    return _to(conn, app_id, ApplicationState.DONE)


def _to(conn, app_id: int, new_state: ApplicationState) -> str:
    db.update_application_state(conn, app_id, new_state.value)
    return new_state.value


def _classify_fields(conn, app_id: int, llm: LLMClient) -> None:
    fields = db.list_fields(conn, app_id)
    result = llm.complete_json(
        system_prompt="Classify each form field as formal or informal, and give a category.",
        user_prompt="classify_fields",
        schema_hint='{"fields": [{"label": str, "category": str, "profile_key": str}]}',
    )
    by_label = {f["label"]: f for f in result["fields"]}
    for row in fields:
        info = by_label.get(row["label"])
        if info is None:
            continue
        conn.execute(
            "UPDATE application_fields SET category = ? WHERE id = ?",
            (info["category"], row["id"]),
        )
    conn.commit()


def _resolve_formal_fields(conn, app_id: int, profile: dict[str, str]) -> None:
    fields = db.list_fields(conn, app_id)
    for row in fields:
        if row["category"] != FORMAL_CATEGORY:
            continue
        key = row["label"].lower().replace(" ", "_")
        value = profile.get(key)
        if value is not None:
            db.update_field(conn, row["id"], resolved_value=value, needs_input=False, resolved_from="profile")
        else:
            db.update_field(conn, row["id"], needs_input=True)


def _resolve_informal_fields(conn, app_id: int, llm: LLMClient, embedder: EmbeddingClient) -> bool:
    fields = db.list_fields(conn, app_id)
    gap = False
    for row in fields:
        if row["category"] == FORMAL_CATEGORY:
            continue
        if row["resolved_value"] is not None:
            continue
        result = memory.retrieve(conn, llm, embedder, row["label"])
        if result["status"] == "needs_input":
            db.update_field(conn, row["id"], needs_input=True, resolved_from=None)
            gap = True
            continue
        answer = memory.synthesize_answer(llm, row["label"], result["candidates"])
        db.update_field(conn, row["id"], resolved_value=answer, needs_input=False, resolved_from="memory")
    return gap


def _draft_cover_letter(conn, app_id: int, llm: LLMClient) -> None:
    result = llm.complete_json(
        system_prompt="Draft a cover letter from the resolved fields.",
        user_prompt="cover letter",
        schema_hint='{"cover_letter": str}',
    )
    conn.execute(
        "UPDATE applications SET cover_letter_draft = ? WHERE id = ?",
        (result["cover_letter"], app_id),
    )
    conn.commit()
