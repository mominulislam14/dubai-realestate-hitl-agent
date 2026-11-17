"""
Graph state definition and Pydantic structured output schemas for DubaiProp HITL.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class DocumentGrade(BaseModel):
    """Structured evaluation of retrieved RAG context relevance."""

    is_relevant: bool = Field(
        description="True if the retrieved documents contain sufficient facts to answer the query."
    )
    reasoning: str = Field(
        description="Brief one-sentence justification for the relevance decision."
    )


class LeadQualification(BaseModel):
    """Structured client qualification extracted from the user inquiry."""

    intent: Literal["general_inquiry", "property_search", "vip_viewing_or_quote"] = Field(
        description="Primary category of the client's inquiry."
    )
    estimated_budget_aed: int | None = Field(
        default=None,
        description="Estimated client budget or target property price in AED if mentioned or inferred; otherwise null.",
    )
    property_interest: str | None = Field(
        default=None,
        description="Specific property type, area, or property ID mentioned (e.g., Palm Jumeirah Villa, JVC Studio).",
    )
    is_high_value: bool = Field(
        description=(
            "True if budget/property value >= 1,000,000 AED, or if the client requests a formal price quote, "
            "custom payment plan, or VIP private viewing."
        )
    )
    qualification_summary: str = Field(
        description="Concise 1-2 sentence executive summary of the client request for the Senior Broker."
    )


class AgentState(TypedDict):
    """Shared state passed across all nodes in the DubaiProp HITL LangGraph workflow."""

    thread_id: str
    user_query: str
    search_query: str
    is_safe: bool
    retrieved_docs: list[dict[str, Any]]
    retrieval_grade: str
    retry_count: int
    lead_info: dict[str, Any]
    draft_response: str
    human_decision: str | None
    human_note: str | None
    final_response: str
    # Reducer: each node appends trace logs rather than overwriting previous steps
    execution_trace: Annotated[list[str], operator.add]