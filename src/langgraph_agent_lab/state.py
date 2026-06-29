"""State schema for the Day 08 LangGraph lab.

Students should extend the schema only when needed. Keep state lean and serializable.
"""

from __future__ import annotations

from enum import StrEnum
from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, field_validator


class Route(StrEnum):
    SIMPLE = "simple"
    TOOL = "tool"
    MISSING_INFO = "missing_info"
    RISKY = "risky"
    ERROR = "error"
    DEAD_LETTER = "dead_letter"
    DONE = "done"


class LabEvent(BaseModel):
    """Append-only audit event for grading and debugging."""

    node: str
    event_type: str
    message: str
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool = False
    reviewer: str = "mock-reviewer"
    comment: str = ""


class AgentState(TypedDict, total=False):
    """LangGraph state.

    Reducer policy:
    - Audit/log-style fields are append-only via ``Annotated[list, add]`` so every
      node contributes to a complete trace (LangGraph merges concurrent updates).
    - Scalar control fields are overwrite (last write wins) because they represent
      the *current* situation, not history.
    """

    # ── Identity / inputs (overwrite) ──────────────────────────────
    thread_id: str
    scenario_id: str
    query: str

    # ── Control / routing scalars (overwrite — current value only) ──
    route: str
    risk_level: str
    attempt: int
    max_attempts: int
    final_answer: str | None

    # ── Student-added fields (overwrite) ───────────────────────────
    # evaluation_result drives the retry-loop gate in route_after_evaluate.
    evaluation_result: str
    # pending_question holds the clarification prompt for missing_info flows.
    pending_question: str
    # proposed_action describes the side-effecting action awaiting approval.
    proposed_action: str
    # approval stores the HITL decision (ApprovalDecision.model_dump()).
    approval: dict[str, Any]

    # ── Append-only audit trails (reducer = operator.add) ──────────
    messages: Annotated[list[str], add]
    tool_results: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    events: Annotated[list[dict[str, Any]], add]


class Scenario(BaseModel):
    id: str
    query: str
    expected_route: Route
    requires_approval: bool = False
    should_retry: bool = False
    max_attempts: int = 3
    tags: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


def initial_state(scenario: Scenario) -> AgentState:
    """Create a serializable initial state for one scenario."""
    return {
        "thread_id": f"thread-{scenario.id}",
        "scenario_id": scenario.id,
        "query": scenario.query,
        "route": "",
        "risk_level": "unknown",
        "attempt": 0,
        "max_attempts": scenario.max_attempts,
        "final_answer": None,
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }


def make_event(node: str, event_type: str, message: str, **metadata: Any) -> dict[str, Any]:
    """Create a normalized event payload."""
    return LabEvent(node=node, event_type=event_type, message=message, metadata=metadata).model_dump()
