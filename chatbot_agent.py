import logging
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from config import get_config

"""
LexGuard — Chatbot Agent

Capabilities:
  1. Search past contract reviews from MongoDB
  2. Explain legal terms (general Q&A)
  3. Suggest redlines based on the current risk register

How it works:
  - Takes user question + current pipeline state + chat history
  - Classifies question intent (current_contract / db_search / legal_qa / redline)
  - Builds a targeted context from the right data source
  - Calls Groq LLM with context + history + question
  - Saves message to MongoDB chat_sessions

LLM   : Llama 3.3 70B via Groq
Memory: MongoDB chat_sessions collection
"""

logger = logging.getLogger(__name__)
LLM_MODEL = "llama-3.3-70b-versatile"


# MAIN — answer a single user message
def answer(
    question: str,
    state,                    # Current PipelineState
    session_id: str,          # Unique chat session ID
    chat_history: list[dict], # [{role, content}, ...] — last N messages
) -> str:
    """
    Main entry point. Takes a question and returns an answer string.
    Also saves the Q&A to MongoDB.
    """
    _load_env()

    client = _get_groq_client()
    if client is None:
        return "Chatbot unavailable — GROQ_API_KEY not set."

    # Step 1 — classify intent
    intent = _classify_intent(question)
    logger.info(f"[Chatbot] Intent: {intent} | Q: {question[:60]}")

    # Step 2 — build context based on intent
    context = _build_context(question, intent, state)

    # Step 3 — build messages for LLM
    system_prompt = _build_system_prompt(state)
    messages = [{"role": "system", "content": system_prompt}]

    # Add last 6 messages of history for memory
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add context + question
    user_content = f"{context}\n\nQuestion: {question}" if context else question
    messages.append({"role": "user", "content": user_content})

    # Step 4 — call Groq
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[Chatbot] LLM call failed: {e}")
        answer_text = f"Sorry, I encountered an error: {e}"

    # Step 5 — save to MongoDB
    try:
        from database import save_message
        save_message(session_id, state.doc_hash or "", "user",      question)
        save_message(session_id, state.doc_hash or "", "assistant", answer_text)
    except Exception as e:
        logger.warning(f"[Chatbot] DB save failed: {e}")

    return answer_text


# INTENT CLASSIFICATION
def _classify_intent(question: str) -> str:
    """
    Rule-based intent classifier — fast, no LLM call needed.

    Intents:
      db_search       — searching past reviews in MongoDB
      redline         — asking for redline / fix suggestions
      legal_qa        — general legal term explanation
      current_contract — default — about current contract
    """
    q = question.lower()

    db_keywords = [
        "past", "previous", "before", "history", "other contract",
        "have we", "did we", "reviewed before", "search", "find contracts",
        "all nda", "all sla", "all vendor", "all partnership",
        "how many contracts", "list contracts",
    ]
    redline_keywords = [
        "redline", "fix", "suggest", "rewrite", "replace", "standard language",
        "what should it say", "better clause", "recommend clause",
        "how to fix", "correct version",
    ]
    legal_keywords = [
        "what is", "what does", "explain", "define", "meaning of",
        "what mean", "legal term", "force majeure", "indemnification",
        "indemnify", "liability cap", "jurisdiction", "arbitration",
        "governing law", "ip ownership", "confidential", "termination clause",
    ]

    if any(k in q for k in db_keywords):
        return "db_search"
    if any(k in q for k in redline_keywords):
        return "redline"
    if any(k in q for k in legal_keywords):
        return "legal_qa"
    return "current_contract"


# CONTEXT BUILDER — per intent
def _build_context(question: str, intent: str, state) -> str:
    """Build the relevant context to inject into the LLM prompt."""

    if intent == "db_search":
        return _context_db_search(question, state)

    if intent == "redline":
        return _context_redline(state)

    if intent == "legal_qa":
        return ""  # LLM general knowledge — no extra context needed

    # Default: current_contract
    return _context_current_contract(state)


def _context_current_contract(state) -> str:
    """Inject current contract metadata + risk register as context."""
    meta = state.contract_metadata
    reg  = state.risk_register

    # Metadata summary
    meta_lines = []
    for k, v in meta.items():
        if k.startswith("_") or v == "Not found":
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        meta_lines.append(f"  {k.replace('_',' ').title()}: {v}")

    # Risk register summary
    high_risks = [r for r in reg if r["severity"] == "HIGH"]
    risk_lines = [
        f"  - [{r['severity']}] {r['canonical_title']}: {r['deviation_summary'][:120]}"
        for r in high_risks[:5]
    ]

    # Clause comparison for detail
    comp_lines = [
        f"  - {c['canonical_title']}: similarity {c['similarity_score']:.0%} "
        f"({'deviated' if c['is_deviated'] else 'aligned'})"
        for c in state.clause_comparisons[:8]
    ]

    return f"""CURRENT CONTRACT CONTEXT:
File: {state.file_name}
Type: {state.contract_type} ({state.contract_type_confidence} confidence)
Pages: {state.page_count}

CONTRACT METADATA:
{chr(10).join(meta_lines) or "  Not available"}

RISK REGISTER (HIGH items):
{chr(10).join(risk_lines) or "  No HIGH risks found"}

CLAUSE COMPARISON RESULTS:
{chr(10).join(comp_lines) or "  Not available"}"""


def _context_redline(state) -> str:
    """Build redline context from deviated clauses + standard texts."""
    deviated = [r for r in state.risk_register if r.get("is_deviated")]
    if not deviated:
        return "No deviations found in this contract — all clauses align with the standard."

    lines = []
    for r in deviated[:5]:
        std = r.get("standard_text", "")[:300]
        lines.append(
            f"CLAUSE: {r['canonical_title']} [{r['severity']}]\n"
            f"  Deviation: {r['deviation_summary']}\n"
            f"  Standard language: {std}\n"
            f"  Recommendation: {r['recommendation']}"
        )

    return (
        f"DEVIATED CLAUSES IN THIS CONTRACT ({len(deviated)} total):\n\n"
        + "\n\n".join(lines)
    )


def _context_db_search(question: str, state) -> str:
    """Query MongoDB for past reviews relevant to the question."""
    try:
        from database import (
            get_recent_reviews, get_reviews_by_type,
            get_high_risk_reviews, text_search_reviews
        )
    except ImportError:
        return "Database module not available."

    q = question.lower()
    reviews = []

    # Decide which DB query to run
    for ctype in ["nda", "sla", "vendor", "partnership"]:
        if ctype in q:
            reviews = get_reviews_by_type(ctype)
            break

    if not reviews and ("high risk" in q or "risky" in q):
        reviews = get_high_risk_reviews()

    if not reviews:
        reviews = get_recent_reviews(limit=5)

    if not reviews:
        return "No past contract reviews found in the database yet."

    lines = []
    for r in reviews[:5]:
        rs = r.get("risk_summary", {})
        meta = r.get("contract_metadata", {})
        party = (
            meta.get("parties") or meta.get("vendor_name") or
            meta.get("partners") or meta.get("service_provider") or "Unknown"
        )
        if isinstance(party, list):
            party = ", ".join(str(p) for p in party[:2])
        lines.append(
            f"  - {r['file_name']} | {r['contract_type']} | "
            f"Reviewed: {r.get('uploaded_at','?')} | "
            f"Risks: {rs.get('high',0)} HIGH, {rs.get('medium',0)} MED | "
            f"Party: {str(party)[:50]}"
        )

    return f"PAST CONTRACT REVIEWS FROM DATABASE ({len(reviews)} found):\n" + "\n".join(lines)


# SYSTEM PROMPT
def _build_system_prompt(state) -> str:
    return f"""You are LexGuard AI, a legal contract review assistant.

You are currently helping review: {state.file_name} ({state.contract_type} contract)

Your capabilities:
1. Answer questions about the current contract being reviewed
2. Search and summarise past contract reviews from the database
3. Explain legal terms and concepts clearly
4. Suggest standard redline language for deviated clauses

Guidelines:
- Be concise, precise, and professional
- For risk questions, always mention severity (HIGH/MEDIUM/LOW)
- For redline suggestions, provide the exact standard clause text
- For legal terms, give a plain English explanation first, then the legal definition
- If you don't know something, say so — never fabricate contract details
- Keep answers under 300 words unless a detailed redline is requested"""


# HELPERS
def _get_groq_client():
    try:
        api_key = get_config("GROQ_API_KEY")
        return Groq(api_key=api_key) if api_key else None
    except ImportError:
        return None


def _load_env():
    try:
        env_path = Path(__file__).resolve().parent / ".env"
        load_dotenv(dotenv_path=env_path) if env_path.exists() else load_dotenv()
    except ImportError:
        pass
