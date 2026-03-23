import json
import logging
import os
import re
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from agent_state import AgentStatus


"""
LexGuard — Metadata Extraction Agent  (Agent 2)

LLM      : Llama 3.3 70B via Groq API
Why Groq : 100% free, no credit card, 14,400 req/day,
           extremely fast inference, great at structured JSON extraction
Model    : llama-3.3-70b-versatile
API Docs : https://console.groq.com
"""

logger = logging.getLogger(__name__)

AGENT_NAME = "MetadataExtractionAgent"
MODEL_NAME = "llama-3.3-70b-versatile"



# METADATA SCHEMAS 
METADATA_SCHEMAS = {
    "NDA": {
        "effective_date":         "The date the agreement becomes effective",
        "parties":                "List of all parties with their roles (e.g. Disclosing Party, Receiving Party)",
        "term":                   "Duration of the agreement (e.g. 2 years from effective date)",
        "jurisdiction":           "Governing law and jurisdiction (e.g. State of Delaware)",
        "confidentiality_period": "How long confidentiality obligations survive termination",
    },
    "SLA": {
        "effective_date":   "The date the agreement becomes effective",
        "service_provider": "Full name of the service provider company",
        "customer":         "Full name of the customer company",
        "service_scope":    "Brief description of the services covered",
        "jurisdiction":     "Governing law and jurisdiction",
    },
    "VENDOR": {
        "effective_date": "The date the agreement becomes effective",
        "vendor_name":    "Full legal name of the vendor",
        "client_name":    "Full legal name of the client",
        "term":           "Duration of the agreement",
        "jurisdiction":   "Governing law and jurisdiction",
        "payment_terms":  "Payment timeline and conditions (e.g. Net 30 days)",
    },
    "PARTNERSHIP": {
        "effective_date":  "The date the agreement becomes effective",
        "partners":        "List of all partners with ownership percentages if stated",
        "business_name":   "Name of the partnership or business entity",
        "jurisdiction":    "Governing law and jurisdiction",
        "term":            "Duration of the partnership",
        "ownership_split": "How ownership or profits are divided among partners",
    },
}

FALLBACK_SCHEMA = {
    "effective_date":  "The date the agreement becomes effective",
    "parties":         "All parties involved in the agreement",
    "term":            "Duration of the agreement",
    "jurisdiction":    "Governing law and jurisdiction",
    "key_obligations": "Main obligations of each party",
}


# MAIN ENTRY POINT
def run(state):
    logger.info(f"[{AGENT_NAME}] Starting → contract type: {state.contract_type}")
    state.metadata_status = AgentStatus.RUNNING

    # Load .env
    _load_env()

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fail(state,
            "GROQ_API_KEY not found. "
            "Get a free key at https://console.groq.com and add to .env: "
            "GROQ_API_KEY=your_key_here")

    # Get schema for this contract type
    schema = METADATA_SCHEMAS.get(state.contract_type, FALLBACK_SCHEMA)
    prompt = _build_prompt(state.clean_text, state.contract_type, schema)

    # Call Groq
    try:
        logger.info(f"[{AGENT_NAME}] Calling {MODEL_NAME} via Groq...")
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal contract analysis assistant. "
                        "You extract structured metadata from contracts. "
                        "You always respond with valid JSON only — no markdown, "
                        "no explanation, no code blocks."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,     # Deterministic — important for structured extraction
            max_tokens=1024,
        )
        raw = response.choices[0].message.content
        logger.info(f"[{AGENT_NAME}] Response received ({len(raw)} chars)")
    except Exception as e:
        return _fail(state, f"Groq API call failed: {e}")

    # Parse JSON
    try:
        metadata = _parse_json(raw)
    except Exception as e:
        return _fail(state, f"JSON parse failed: {e} | Raw: {raw[:300]}")

    # Fill missing fields
    for field in schema:
        if field not in metadata:
            metadata[field] = "Not found"

    # Add internal tracking fields
    metadata["_contract_type"] = state.contract_type
    metadata["_model"]         = MODEL_NAME
    metadata["_schema_fields"] = list(schema.keys())

    state.contract_metadata = metadata
    state.metadata_status   = AgentStatus.COMPLETED
    logger.info(f"[{AGENT_NAME}] Done — {len(schema)} fields extracted.")
    return state


# HELPERS
def _load_env():
    try:
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"[{AGENT_NAME}] Loaded .env from {env_path}")
        else:
            load_dotenv()
    except ImportError:
        pass


def _build_prompt(text, contract_type, schema):
    field_list = "\n".join(
        f'  "{k}": "{v}"' for k, v in schema.items()
    )
    doc_text = text[:6000] if len(text) > 6000 else text

    return f"""Extract the following metadata fields from this {contract_type} contract.

Return ONLY a valid JSON object with exactly these keys:
{{
{field_list}
}}

Rules:
- If a field cannot be found, use "Not found"
- For list fields (parties, partners) return a JSON array of strings
- Return raw JSON only — no markdown, no code blocks, no explanation

CONTRACT TEXT:
{doc_text}
"""


def _parse_json(raw):
    text = raw.strip()
    # Strip markdown code fences if model adds them
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$',          '', text, flags=re.MULTILINE)
    return json.loads(text.strip())


def _fail(state, error):
    logger.error(f"[{AGENT_NAME}] FAILED — {error}")
    state.metadata_status = AgentStatus.FAILED
    state.metadata_error  = error
    return state