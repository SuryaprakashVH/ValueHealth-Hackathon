import json
import logging
import os
import re
from pathlib import Path
from agent_state import AgentStatus
from groq import Groq
from dotenv import load_dotenv

"""
LexGuard — Risk Classification Agent

Role         : Read every deviated clause from Agent 3+4 and classify
               each one with severity, risk type, business impact,
               and a concrete recommendation.

Responsibility:
    - Read state.clause_comparisons (from Agent 3+4)
    - For each clause where is_deviated = True:
        * Ask Groq LLM to classify severity (LOW / MEDIUM / HIGH)
        * Ask Groq LLM to identify risk type
        * Ask Groq LLM to explain business impact in plain English
        * Ask Groq LLM to suggest what the legal team should do
    - For clauses where is_deviated = False → mark as ACCEPTED (no risk)
    - Build state.risk_register — the full risk table for the report

Why LLM here?
    Rule-based scoring (e.g. "if similarity < 0.5 = HIGH") is too blunt.
    The same similarity score means different things for different clause
    types — a 60% similarity on a LIABILITY CAP clause is HIGH risk,
    but a 60% on a JURISDICTION clause may only be LOW risk.
    The LLM understands legal context and can reason about severity
    the way a junior lawyer would.

LLM Used : Llama 3.3 70B via Groq
Input    : state.clause_comparisons (from Agent 3+4)
Output   : state.risk_register — list of classified risk items
"""

logger = logging.getLogger(__name__)

AGENT_NAME = "RiskClassificationAgent"
LLM_MODEL  = "llama-3.3-70b-versatile"


# ══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def run(state):
    logger.info(f"[{AGENT_NAME}] Starting → {len(state.clause_comparisons)} comparisons to classify")
    state.risk_status = AgentStatus.RUNNING

    _load_env()

    # Get Groq client
    groq_client = _get_groq_client()
    if groq_client is None:
        return _fail(state, "GROQ_API_KEY not set or groq package not installed.")

    risk_register = []

    for comp in state.clause_comparisons:

        if not comp.get("is_deviated"):
            # ── Clause is aligned — no risk ───────────────────────────
            risk_register.append({
                "clause_id":        comp.get("clause_id", ""),
                "canonical_title":  comp["canonical_title"],
                "category":         comp["category"],
                "severity":         "ACCEPTED",
                "similarity_score": comp["similarity_score"],
                "deviation_summary":comp["deviation_summary"],
                "business_impact":  "Clause aligns with the standard. No action required.",
                "recommendation":   "Accept as-is.",
                "standard_text":    comp.get("standard_text", ""),
                "contract_text":    comp.get("contract_text", ""),
                "is_deviated":      False,
            })
            logger.info(f"[{AGENT_NAME}] {comp['canonical_title']} → ACCEPTED")
            continue

        # ── Clause is deviated — classify with LLM ────────────────────
        classification = _classify_with_llm(
            client          = groq_client,
            canonical_title = comp["canonical_title"],
            category        = comp["category"],
            contract_text   = comp.get("contract_text", ""),
            standard_text   = comp.get("standard_text", ""),
            similarity      = comp["similarity_score"],
            deviation_summary = comp["deviation_summary"],
        )

        risk_register.append({
            "clause_id":         comp.get("clause_id", ""),
            "canonical_title":   comp["canonical_title"],
            "category":          comp["category"],
            "severity":          classification.get("severity",       "MEDIUM"),
            "business_impact":   classification.get("business_impact", ""),
            "recommendation":    classification.get("recommendation",  ""),
            "similarity_score":  comp["similarity_score"],
            "deviation_summary": comp["deviation_summary"],
            "standard_text":     comp.get("standard_text", ""),
            "contract_text":     comp.get("contract_text", ""),
            "is_deviated":       True,
        })

        logger.info(
            f"[{AGENT_NAME}] {comp['canonical_title']} → "
            f"{classification.get('severity','?')}"
        )

    state.risk_register = risk_register
    state.risk_status   = AgentStatus.COMPLETED

    high   = sum(1 for r in risk_register if r["severity"] == "HIGH")
    medium = sum(1 for r in risk_register if r["severity"] == "MEDIUM")
    low    = sum(1 for r in risk_register if r["severity"] == "LOW")
    logger.info(
        f"[{AGENT_NAME}] Done — HIGH: {high}, MEDIUM: {medium}, LOW: {low}, "
        f"ACCEPTED: {len(risk_register) - high - medium - low}"
    )
    return state

# LLM CLASSIFICATION
def _classify_with_llm(client, canonical_title, category, contract_text,
                        standard_text, similarity, deviation_summary):
    """
    Ask Groq LLM to classify a deviated clause.
    Returns a dict with: severity, risk_type, business_impact, recommendation
    """

    prompt = f"""You are a senior legal risk analyst reviewing a contract deviation.

A clause in this contract deviates from the standard template.
Your job is to classify the risk and provide actionable guidance.

CLAUSE NAME     : {canonical_title}
CLAUSE CATEGORY : {category}
SIMILARITY SCORE: {similarity:.0%} (lower = more different from standard)
DEVIATION SUMMARY: {deviation_summary}

STANDARD CLAUSE:
{standard_text}

CONTRACT CLAUSE (what the document actually says):
{contract_text[:500]}

Classify this deviation and return ONLY a valid JSON object:
{{
  "severity": "HIGH" or "MEDIUM" or "LOW",
  "business_impact": "1-2 sentences explaining the real-world business/legal consequence if this clause is accepted as-is",
  "recommendation": "1 concrete action the legal team should take (e.g. negotiate, reject, add clause, etc.)"
}}

Severity guide:
- HIGH   = exposes company to significant financial liability, IP loss, or legal action
- MEDIUM = non-standard terms that need negotiation but are not immediately dangerous
- LOW    = minor wording differences with minimal practical impact

Return raw JSON only — no markdown, no explanation."""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior legal risk analyst. "
                        "You always respond with valid JSON only."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=400,
        )
        raw = response.choices[0].message.content
        return _parse_json(raw)
    except Exception as e:
        logger.warning(f"[{AGENT_NAME}] LLM call failed for {canonical_title}: {e}")
        return {
            "severity":        "MEDIUM",
            "risk_type":       category,
            "business_impact": f"Deviation detected (similarity: {similarity:.0%}). Manual review required.",
            "recommendation":  "Flag for legal team review.",
        }

# HELPERS
def _get_groq_client():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            return Groq(api_key=api_key)
    except ImportError:
        pass
    return None


def _load_env():
    try:
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
    except ImportError:
        pass


def _parse_json(raw):
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$',          '', text, flags=re.MULTILINE)
    return json.loads(text.strip())


def _fail(state, error):
    logger.error(f"[{AGENT_NAME}] FAILED — {error}")
    state.risk_status = AgentStatus.FAILED
    state.risk_error  = error
    return state