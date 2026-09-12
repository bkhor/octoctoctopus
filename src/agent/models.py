from dataclasses import dataclass, field
from enum import Enum


class ApplicationState(str, Enum):
    QUEUED = "queued"
    FETCHING = "fetching"
    CLASSIFYING = "classifying"
    RESOLVING_FORMAL = "resolving_formal"
    RESOLVING_INFORMAL = "resolving_informal"
    AWAITING_HITL = "awaiting_hitl"
    DRAFTING_COVER_LETTER = "drafting_cover_letter"
    FILLING = "filling"
    READY_FOR_REVIEW = "ready_for_review"
    DONE = "done"
    MANUAL_FALLBACK = "manual_fallback"


@dataclass
class FieldSpec:
    label: str
    field_type: str
    options: list[str] = field(default_factory=list)
    required: bool = False
