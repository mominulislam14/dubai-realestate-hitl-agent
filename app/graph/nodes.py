"""
LangGraph node implementations for DubaiProp HITL.
Each node reads from AgentState, performs a focused step, and returns partial state updates.
"""

import asyncio
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.types import interrupt

from app.core.config import settings
from app.graph.state import AgentState, DocumentGrade, LeadQualification
from app.rag.vector_store import search_knowledge_base

_llm: ChatGroq | None = None

# Fast deterministic prompt-injection patterns
INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|above|prior) instructions",
    r"reveal (your )?(system prompt|instructions|rules)",
    r"you are now (dan|developer mode|jailbroken)",
    r"bypass (all )?(security|guardrails|approval)",
    r"override system",
]


def get_llm() -> ChatGroq:
    """Return a lazily initialized singleton ChatGroq client."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )
    return _llm


async def guardrail_node(state: AgentState) -> dict[str, Any]:
    """Validate input safety and block prompt-injection attempts deterministically."""
    query = (state.get("user_query") or "").strip()
    if not query:
        return {
            "is_safe": False,
            "execution_trace": ["guardrail_node -> BLOCKED (empty query)"],
        }

    lowered = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return {
                "is_safe": False,
                "execution_trace": ["guardrail_node -> BLOCKED (prompt injection detected)"],
            }

    return {
        "is_safe": True,
        "search_query": query,
        "retry_count": 0,
        "execution_trace": ["guardrail_node -> PASSED"],
    }


async def block_node(state: AgentState) -> dict[str, Any]:
    """Return a safe refusal response when guardrail blocks the request."""
    return {
        "final_response": (
            "I cannot process this request due to security policy restrictions. "
            "Please ask a valid question regarding Dubai real estate properties or investment policies."
        ),
        "execution_trace": ["block_node -> Returned safe security refusal"],
    }


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Retrieve top relevant documents from Qdrant vector store."""
    query = state.get("search_query") or state["user_query"]
    docs = await asyncio.to_thread(search_knowledge_base, query, 3)

    top_info = f"{docs[0]['title']} (score: {docs[0]['score']})" if docs else "no results"
    return {
        "retrieved_docs": docs,
        "execution_trace": [f"retrieve_node -> Retrieved {len(docs)} docs | Top: {top_info}"],
    }


async def grade_docs_node(state: AgentState) -> dict[str, Any]:
    """Evaluate whether retrieved documents actually answer the user query (Corrective RAG)."""
    docs = state.get("retrieved_docs") or []
    user_query = state["user_query"]

    # Fast cutoff if vector similarity is clearly too low
    if not docs or max(d.get("score", 0.0) for d in docs) < 0.25:
        return {
            "retrieval_grade": "irrelevant",
            "execution_trace": ["grade_docs_node -> IRRELEVANT (low vector similarity)"],
        }

    context_preview = "\n\n".join(
        f"[{d['title']}]: {d['content']}" for d in docs
    )

    system_msg = SystemMessage(
        content=(
            "You are a strict relevance grader for a Dubai real estate AI system. "
            "Determine whether the retrieved documents contain factual information directly useful "
            "to answer the user's question. Be strict: if the user asks about something completely "
            "unrelated or outside the provided documents, mark is_relevant as false."
        )
    )
    human_msg = HumanMessage(
        content=f"User Question: {user_query}\n\nRetrieved Documents:\n{context_preview}"
    )

    try:
        structured_llm = get_llm().with_structured_output(DocumentGrade)
        grade: DocumentGrade = await structured_llm.ainvoke([system_msg, human_msg])
        status = "relevant" if grade.is_relevant else "irrelevant"
        reason = grade.reasoning
    except Exception:
        # Fail-safe fallback based on vector score
        status = "relevant" if docs[0].get("score", 0.0) >= 0.45 else "irrelevant"
        reason = "Fallback vector threshold check"

    return {
        "retrieval_grade": status,
        "execution_trace": [f"grade_docs_node -> {status.upper()} ({reason})"],
    }


async def rewrite_query_node(state: AgentState) -> dict[str, Any]:
    """Rewrite the search query using domain keywords to improve vector retrieval."""
    current_query = state.get("search_query") or state["user_query"]
    next_retry = state.get("retry_count", 0) + 1

    system_msg = SystemMessage(
        content=(
            "You are a search query optimizer for a Dubai real estate knowledge base "
            "(covering DLD fees, UAE Golden Visa, RERA Escrow, and Dubai properties). "
            "Rewrite the user's question into a concise, keyword-rich search query to improve "
            "semantic vector retrieval. Output ONLY the rewritten query text."
        )
    )
    human_msg = HumanMessage(content=f"Original Query: {current_query}")

    try:
        response = await get_llm().ainvoke([system_msg, human_msg])
        rewritten = response.content.strip().strip('"') or current_query
    except Exception:
        rewritten = current_query

    return {
        "search_query": rewritten,
        "retry_count": next_retry,
        "execution_trace": [f"rewrite_query_node -> Attempt #{next_retry}: '{rewritten}'"],
    }


async def qualify_lead_node(state: AgentState) -> dict[str, Any]:
    """Extract structured lead metadata and enforce high-value HITL governance rules."""
    user_query = state["user_query"]
    docs = state.get("retrieved_docs") or []
    context_summary = "\n".join(f"- {d['title']}: {d['content']}" for d in docs)

    system_msg = SystemMessage(
        content=(
            "You are a Senior Dubai Real Estate Lead Qualification Analyst. "
            "Extract structured client intent, estimated budget or matched property price in AED, "
            "and determine if this inquiry requires Senior Broker approval.\n"
            "Governance Rule: Set is_high_value=True if the budget or matched property price is >= 1,000,000 AED, "
            "OR if the client requests a formal price quote, custom payment plan, or VIP private viewing."
        )
    )
    human_msg = HumanMessage(
        content=f"Client Inquiry: {user_query}\n\nMatched Knowledge Context:\n{context_summary}"
    )

    try:
        structured_llm = get_llm().with_structured_output(LeadQualification)
        lead: LeadQualification = await structured_llm.ainvoke([system_msg, human_msg])
        lead_dict = lead.model_dump()
    except Exception:
        lead_dict = {
            "intent": "general_inquiry",
            "estimated_budget_aed": None,
            "property_interest": None,
            "is_high_value": False,
            "qualification_summary": "General client inquiry.",
        }

    # Deterministic policy enforcement (never rely solely on LLM judgment)
    budget = lead_dict.get("estimated_budget_aed")
    if isinstance(budget, (int, float)) and budget >= settings.HIGH_VALUE_THRESHOLD_AED:
        lead_dict["is_high_value"] = True
    if lead_dict.get("intent") == "vip_viewing_or_quote":
        lead_dict["is_high_value"] = True

    flag_label = "HIGH-VALUE (HITL Required)" if lead_dict["is_high_value"] else "STANDARD (Auto-Reply)"
    return {
        "lead_info": lead_dict,
        "execution_trace": [f"qualify_lead_node -> {flag_label} | Intent: {lead_dict['intent']}"],
    }


async def draft_action_node(state: AgentState) -> dict[str, Any]:
    """Draft a grounded, professional real estate response or formal deal proposal."""
    user_query = state["user_query"]
    grade = state.get("retrieval_grade", "relevant")
    docs = state.get("retrieved_docs") or []
    lead_info = state.get("lead_info") or {}

    if grade != "relevant":
        draft = (
            "Thank you for reaching out to our Dubai Real Estate Advisory Desk. "
            "I could not find verified information in our current brokerage database to answer your specific query accurately. "
            "I am forwarding your request to a Senior Property Advisor who will assist you directly."
        )
        return {
            "draft_response": draft,
            "execution_trace": ["draft_action_node -> Generated grounded fallback (no hallucination)"],
        }

    context_text = "\n\n".join(f"### {d['title']}\n{d['content']}" for d in docs)
    is_high_value = lead_info.get("is_high_value", False)

    tone_instruction = (
        "This is a HIGH-VALUE inquiry. Draft a concise executive summary (under 180 words) covering the matched property, "
        "exact AED pricing, Golden Visa eligibility (if applicable), and next steps for Senior Broker viewing confirmation."
        if is_high_value
        else "Draft a clear, concise advisory response (under 120 words) grounded strictly in the provided context."
    )

    system_msg = SystemMessage(
        content=(
            "You are a professional Dubai Real Estate AI Advisor representing DubaiProp HITL Advisory Desk. "
            "Answer strictly using the Verified Context below. Never invent prices, fees, or property details "
            "that are not in the context.\n"
            "Formatting Rules: Use short bullet points instead of markdown pipe tables (never output '|' tables). "
            "Address the user directly without bracket placeholders like [Client Name], [Your Name], or [Brokerage Name]. "
            "If signing off, sign simply as 'DubaiProp HITL Dubai Real Estate Advisory Desk'.\n\n"
            f"{tone_instruction}\n\nVerified Context:\n{context_text}"
        )
    )

    human_msg = HumanMessage(content=user_query)

    response = await get_llm().ainvoke([system_msg, human_msg])
    return {
        "draft_response": response.content.strip(),
        "execution_trace": ["draft_action_node -> Drafted grounded response"],
    }


async def human_approval_node(state: AgentState) -> dict[str, Any]:
    """
    Pause graph execution for Human-in-the-Loop (HITL) Senior Broker approval.
    State is persisted via Checkpointer until resumed with Command(resume=...).
    """
    decision_payload = interrupt(
        {
            "message": "Senior Broker authorization required before dispatching high-value response.",
            "lead_info": state.get("lead_info", {}),
            "draft_response": state.get("draft_response", ""),
        }
    )

    decision = "rejected"
    note = ""
    if isinstance(decision_payload, dict):
        decision = str(decision_payload.get("decision", "rejected")).lower()
        note = str(decision_payload.get("note", "")).strip()
    elif isinstance(decision_payload, str):
        decision = decision_payload.lower()

    return {
        "human_decision": decision,
        "human_note": note,
        "execution_trace": [f"human_approval_node -> Resumed with human decision: {decision.upper()}"],
    }


async def finalize_node(state: AgentState) -> dict[str, Any]:
    """Produce the final client-facing response based on auto-approval or human decision."""
    human_decision = state.get("human_decision")
    human_note = (state.get("human_note") or "").strip()
    draft = state.get("draft_response", "")

    if human_decision == "rejected":
        reason_text = f" Broker Note: {human_note}" if human_note else ""
        return {
            "final_response": (
                "Your inquiry has been reviewed by our Senior Brokerage Desk and requires custom offline handling. "
                f"A Senior Advisor will contact you shortly.{reason_text}"
            ),
            "execution_trace": ["finalize_node -> Rejected by Senior Broker (held from auto-dispatch)"],
        }

    final_text = draft
    if human_decision == "approved" and human_note:
        final_text = f"{draft}\n\n---\n**Senior Broker Note:** {human_note}"

    mode = "Human-Approved" if human_decision == "approved" else "Auto-Dispatched"
    return {
        "final_response": final_text,
        "execution_trace": [f"finalize_node -> {mode} final response"],
    }