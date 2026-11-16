"""
LangGraph StateGraph construction and conditional routing for SentinelOps.
Connects guardrails, corrective RAG cycles, lead qualification, and HITL interrupts.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.graph.nodes import (
    block_node,
    draft_action_node,
    finalize_node,
    grade_docs_node,
    guardrail_node,
    human_approval_node,
    qualify_lead_node,
    retrieve_node,
    rewrite_query_node,
)
from app.graph.state import AgentState


def route_after_guardrail(state: AgentState) -> Literal["retrieve", "block"]:
    """Route safe queries to retrieval and unsafe queries to block node."""
    return "retrieve" if state.get("is_safe", False) else "block"


def route_after_grade(state: AgentState) -> Literal["qualify_lead", "rewrite_query"]:
    """
    Corrective RAG router:
    Proceed if documents are relevant OR max retry attempts reached;
    otherwise loop back to rewrite the query.
    """
    grade = state.get("retrieval_grade", "irrelevant")
    retries = state.get("retry_count", 0)

    if grade == "relevant" or retries >= settings.MAX_RAG_RETRIES:
        return "qualify_lead"
    return "rewrite_query"


def route_after_draft(state: AgentState) -> Literal["human_approval", "finalize"]:
    """
    HITL governance router:
    Pause for Senior Broker approval if inquiry is flagged as high-value;
    otherwise auto-dispatch via finalize node.
    """
    lead_info = state.get("lead_info") or {}
    if lead_info.get("is_high_value", False):
        return "human_approval"
    return "finalize"


def build_workflow() -> StateGraph:
    """Build and wire the SentinelOps LangGraph workflow (uncompiled)."""
    builder = StateGraph(AgentState)

    # Register all nodes
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("block", block_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_docs", grade_docs_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("qualify_lead", qualify_lead_node)
    builder.add_node("draft_action", draft_action_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("finalize", finalize_node)

    # Entry & guardrail branch
    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"retrieve": "retrieve", "block": "block"},
    )
    builder.add_edge("block", END)

    # Corrective RAG cycle
    builder.add_edge("retrieve", "grade_docs")
    builder.add_conditional_edges(
        "grade_docs",
        route_after_grade,
        {"qualify_lead": "qualify_lead", "rewrite_query": "rewrite_query"},
    )
    builder.add_edge("rewrite_query", "retrieve")

    # Lead qualification & HITL branch
    builder.add_edge("qualify_lead", "draft_action")
    builder.add_conditional_edges(
        "draft_action",
        route_after_draft,
        {"human_approval": "human_approval", "finalize": "finalize"},
    )
    builder.add_edge("human_approval", "finalize")
    builder.add_edge("finalize", END)

    return builder


if __name__ == "__main__":
    import asyncio
    import uuid
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command
    from app.rag.vector_store import get_qdrant_client

    async def _run_cli_smoke_test() -> None:
        workflow = build_workflow()
        async with AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH) as checkpointer:
            await checkpointer.setup()
            graph = workflow.compile(checkpointer=checkpointer)

            print("\n=== TEST 1: Standard Policy FAQ (Auto-Dispatch) ===")
            t1 = {"configurable": {"thread_id": f"test-faq-{uuid.uuid4().hex[:6]}"}}
            res1 = await graph.ainvoke(
                {"user_query": "What is the mandatory DLD transfer fee in Dubai?"},
                config=t1,
            )
            print("Trace:", " | ".join(res1["execution_trace"]))
            print("Final Response:\n", res1["final_response"])

            print("\n=== TEST 2: High-Value VIP Inquiry (HITL Pause -> Resume) ===")
            t2 = {"configurable": {"thread_id": f"test-vip-{uuid.uuid4().hex[:6]}"}}
            res2 = await graph.ainvoke(
                {"user_query": "My budget is 24.5 million AED. I want to book a private VIP viewing for the Palm Jumeirah beachfront villa."},
                config=t2,
            )
            print("Paused at Interrupt?:", "__interrupt__" in res2)
            if "__interrupt__" in res2:
                print("Interrupt Payload:", res2["__interrupt__"][0].value["message"])
                print("Resuming with Senior Broker APPROVAL...")
                res2_final = await graph.ainvoke(
                    Command(resume={"decision": "approved", "note": "VIP viewing confirmed for Thursday 4:00 PM."}),
                    config=t2,
                )
                print("Trace:", " | ".join(res2_final["execution_trace"]))
                print("Final Response:\n", res2_final["final_response"])

    try:
        asyncio.run(_run_cli_smoke_test())
    finally:
        get_qdrant_client().close()