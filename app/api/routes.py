"""
FastAPI routes for DubaiProp HITL agent execution, HITL resume, and state inspection.
"""

import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.config import settings

router = APIRouter(prefix="/api", tags=["DubaiProp HITL Agent"])


class InquiryRequest(BaseModel):
    """Incoming client inquiry payload."""

    query: str = Field(..., min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, max_length=100)


class ApprovalRequest(BaseModel):
    """Human-in-the-Loop Senior Broker decision payload."""

    thread_id: str = Field(..., min_length=1, max_length=100)
    decision: Literal["approved", "rejected"]
    note: str = Field(default="", max_length=500)


class AgentExecutionResponse(BaseModel):
    """Unified response schema for inquiry submission and HITL resume."""

    thread_id: str
    status: Literal["completed", "awaiting_approval"]
    final_response: str | None = None
    draft_response: str | None = None
    lead_info: dict[str, Any] = Field(default_factory=dict)
    retrieved_docs: list[dict[str, Any]] = Field(default_factory=list)
    execution_trace: list[str] = Field(default_factory=list)
    interrupt_message: str | None = None


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Service health endpoint."""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model": settings.GROQ_MODEL,
    }


@router.post("/inquiry", response_model=AgentExecutionResponse)
async def submit_inquiry(payload: InquiryRequest, request: Request) -> AgentExecutionResponse:
    """Run a client inquiry through the LangGraph Corrective RAG & HITL pipeline."""
    graph = request.app.state.graph
    thread_id = (payload.thread_id or f"deal-{uuid.uuid4().hex[:8]}").strip()
    config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(
        {"user_query": payload.query.strip(), "thread_id": thread_id},
        config=config,
    )

    if "__interrupt__" in result and result["__interrupt__"]:
        interrupt_data = result["__interrupt__"][0].value
        return AgentExecutionResponse(
            thread_id=thread_id,
            status="awaiting_approval",
            draft_response=interrupt_data.get("draft_response") or result.get("draft_response"),
            lead_info=interrupt_data.get("lead_info") or result.get("lead_info", {}),
            retrieved_docs=result.get("retrieved_docs", []),
            execution_trace=result.get("execution_trace", []),
            interrupt_message=interrupt_data.get("message", "Senior Broker approval required."),
        )

    return AgentExecutionResponse(
        thread_id=thread_id,
        status="completed",
        final_response=result.get("final_response", ""),
        draft_response=result.get("draft_response"),
        lead_info=result.get("lead_info", {}),
        retrieved_docs=result.get("retrieved_docs", []),
        execution_trace=result.get("execution_trace", []),
    )


@router.post("/approve", response_model=AgentExecutionResponse)
async def submit_human_approval(payload: ApprovalRequest, request: Request) -> AgentExecutionResponse:
    """Resume a paused HITL thread from SQLite checkpointer with Senior Broker decision."""
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": payload.thread_id}}

    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Thread not found.")
    if not snapshot.next:
        raise HTTPException(status_code=400, detail="Thread is already completed and not awaiting approval.")

    result = await graph.ainvoke(
        Command(resume={"decision": payload.decision, "note": payload.note}),
        config=config,
    )

    return AgentExecutionResponse(
        thread_id=payload.thread_id,
        status="completed",
        final_response=result.get("final_response", ""),
        draft_response=result.get("draft_response"),
        lead_info=result.get("lead_info", {}),
        retrieved_docs=result.get("retrieved_docs", []),
        execution_trace=result.get("execution_trace", []),
    )


@router.get("/thread/{thread_id}", response_model=AgentExecutionResponse)
async def get_thread_state(thread_id: str, request: Request) -> AgentExecutionResponse:
    """Inspect persisted state of any thread (supports recovery after page reload)."""
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Thread not found.")

    vals = snapshot.values
    is_paused = bool(snapshot.next)

    return AgentExecutionResponse(
        thread_id=thread_id,
        status="awaiting_approval" if is_paused else "completed",
        final_response=vals.get("final_response") if not is_paused else None,
        draft_response=vals.get("draft_response"),
        lead_info=vals.get("lead_info", {}),
        retrieved_docs=vals.get("retrieved_docs", []),
        execution_trace=vals.get("execution_trace", []),
        interrupt_message="Senior Broker approval required." if is_paused else None,
    )