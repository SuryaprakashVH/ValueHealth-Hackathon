import hashlib
import re
import logging
from agent_state import PipelineState, AgentStatus, PageData
import fitz

"""
LexGuard — Document Ingestion Agent

All canonical titles, raw headings, categories and contract signatures
are hardcoded directly from the 4 clause libraries.
No JSON files are loaded at runtime.

Step 1 — PDF Extraction           : PyMuPDF page-by-page
Step 2 — Cleaning & Normalisation : whitespace, unicode, page numbers
Step 3 — Contract Type Detection  : keyword scoring (hardcoded signatures)
Step 4 — Clause Structure Detection : heading pattern matching
Step 5 — Clause Segmentation      : match headings → canonical titles

"""

logger = logging.getLogger(__name__)

AGENT_NAME        = "DocumentIngestionAgent"
MAX_SIZE_MB       = 20
SCANNED_THRESHOLD = 50


CONTRACT_SIGNATURES = {
    "NDA": {
        "primary": [
            r"\bmutual non-disclosure agreement\b",
            r"\bnda\b",
            r"\bconfidentiality agreement\b",
        ],
        "secondary": [
            r"\bconfidential information\b",
            r"\breceiving party\b",
            r"\bdisclosing party\b",
            r"\bproprietary information\b",
            r"\btrade secrets\b",
            r"\bnon-disclosure\b",
        ],
    },
    "SLA": {
        "primary": [
            r"\bservice level agreement\b",
            r"\bsla\b",
            r"\bservice availability agreement\b",
        ],
        "secondary": [
            r"\bservice level\b",
            r"\buptime\b",
            r"\bavailability\b",
            r"\bservice credits\b",
            r"\bdowntime\b",
            r"\bresponse time\b",
            r"\bresolution time\b",
        ],
    },
    "VENDOR": {
        "primary": [
            r"\bvendor agreement\b",
            r"\bservices agreement\b",
            r"\bmaster services agreement\b",
            r"\bmsa\b",
        ],
        "secondary": [
            r"\bvendor\b",
            r"\bservices\b",
            r"\bpayment terms\b",
            r"\bdeliverables\b",
            r"\bindemnity\b",
            r"\bliability\b",
            r"\btermination\b",
            r"\bintellectual property\b",
        ],
    },
    "PARTNERSHIP": {
        "primary": [
            r"\bpartnership agreement\b",
            r"\bbusiness partnership agreement\b",
            r"\bjoint venture agreement\b",
        ],
        "secondary": [
            r"\bpartnership\b",
            r"\bpartners\b",
            r"\bcapital contribution\b",
            r"\bprofit sharing\b",
            r"\bmanagement\b",
            r"\bdissolution\b",
            r"\bownership interest\b",
        ],
    },
}

CLAUSE_LIBRARY = {
    "NDA": [
        {"canonical_title": "Confidential Information Definition",  "raw_heading": "Definition of Confidential Information", "category": "confidentiality", "risk_weight": "HIGH"},
        {"canonical_title": "Confidentiality Obligations",          "raw_heading": "Obligations of Receiving Party",         "category": "confidentiality", "risk_weight": "HIGH"},
        {"canonical_title": "Disclosure to Representatives",        "raw_heading": "Permitted Disclosures",                  "category": "confidentiality", "risk_weight": "HIGH"},
        {"canonical_title": "Exclusions from Confidential Information", "raw_heading": "Exclusions",                         "category": "confidentiality", "risk_weight": "MEDIUM"},
        {"canonical_title": "Legal Disclosure",                     "raw_heading": "Compelled Disclosure",                   "category": "legal",           "risk_weight": "HIGH"},
        {"canonical_title": "Protection Standard",                  "raw_heading": "Standard of Care",                       "category": "security",        "risk_weight": "HIGH"},
        {"canonical_title": "Confidentiality Term",                 "raw_heading": "Term and Survival",                      "category": "term",            "risk_weight": "HIGH"},
        {"canonical_title": "Return of Information",                "raw_heading": "Return or Destruction",                  "category": "data_handling",   "risk_weight": "HIGH"},
        {"canonical_title": "Injunctive Relief",                    "raw_heading": "Remedies",                               "category": "legal_remedy",    "risk_weight": "HIGH"},
        {"canonical_title": "Jurisdiction",                         "raw_heading": "Governing Law",                          "category": "jurisdiction",    "risk_weight": "MEDIUM"},
    ],
    "SLA": [
        {"canonical_title": "Uptime Commitment",                    "raw_heading": "Service Availability",                   "category": "performance",     "risk_weight": "HIGH"},
        {"canonical_title": "Maintenance Window",                   "raw_heading": "Scheduled Maintenance",                  "category": "operations",      "risk_weight": "MEDIUM"},
        {"canonical_title": "Penalty for Downtime",                 "raw_heading": "Service Credits",                        "category": "penalty",         "risk_weight": "HIGH"},
        {"canonical_title": "Response Time SLA",                    "raw_heading": "Incident Response Time",                 "category": "support",         "risk_weight": "HIGH"},
        {"canonical_title": "Resolution SLA",                       "raw_heading": "Resolution Time",                        "category": "support",         "risk_weight": "HIGH"},
        {"canonical_title": "Exclusions from SLA",                  "raw_heading": "Excused Downtime",                       "category": "exclusions",      "risk_weight": "HIGH"},
        {"canonical_title": "Service Monitoring",                   "raw_heading": "Monitoring and Reporting",               "category": "monitoring",      "risk_weight": "MEDIUM"},
        {"canonical_title": "Termination Rights",                   "raw_heading": "Termination for Chronic Failure",        "category": "termination",     "risk_weight": "HIGH"},
        {"canonical_title": "Liability Cap",                        "raw_heading": "Limitation of Liability",                "category": "liability",       "risk_weight": "HIGH"},
        {"canonical_title": "Jurisdiction",                         "raw_heading": "Governing Law",                          "category": "jurisdiction",    "risk_weight": "MEDIUM"},
    ],
    "VENDOR": [
        {"canonical_title": "Service Scope",                        "raw_heading": "Scope of Services",                      "category": "services",        "risk_weight": "HIGH"},
        {"canonical_title": "Payment Terms",                        "raw_heading": "Fees and Payment Terms",                 "category": "payment",         "risk_weight": "HIGH"},
        {"canonical_title": "Service Level Commitment",             "raw_heading": "Service Levels",                         "category": "performance",     "risk_weight": "HIGH"},
        {"canonical_title": "Confidentiality Obligations",          "raw_heading": "Confidentiality",                        "category": "confidentiality", "risk_weight": "HIGH"},
        {"canonical_title": "IP Ownership",                         "raw_heading": "Intellectual Property",                  "category": "ip",              "risk_weight": "HIGH"},
        {"canonical_title": "Vendor Indemnity",                     "raw_heading": "Indemnification",                        "category": "liability",       "risk_weight": "HIGH"},
        {"canonical_title": "Liability Cap",                        "raw_heading": "Limitation of Liability",                "category": "liability",       "risk_weight": "HIGH"},
        {"canonical_title": "Termination Rights",                   "raw_heading": "Termination",                            "category": "termination",     "risk_weight": "HIGH"},
        {"canonical_title": "Regulatory Compliance",                "raw_heading": "Compliance with Laws",                   "category": "compliance",      "risk_weight": "HIGH"},
        {"canonical_title": "Jurisdiction",                         "raw_heading": "Governing Law",                          "category": "jurisdiction",    "risk_weight": "MEDIUM"},
    ],
    "PARTNERSHIP": [
        {"canonical_title": "Formation of Partnership",             "raw_heading": "Formation",                              "category": "structure",              "risk_weight": "HIGH"},
        {"canonical_title": "Capital Contribution",                 "raw_heading": "Capital Contributions",                  "category": "financial",              "risk_weight": "HIGH"},
        {"canonical_title": "Profit Distribution",                  "raw_heading": "Profit and Loss Sharing",                "category": "financial",              "risk_weight": "HIGH"},
        {"canonical_title": "Governance Structure",                 "raw_heading": "Management and Control",                 "category": "governance",             "risk_weight": "HIGH"},
        {"canonical_title": "Reserved Matters",                     "raw_heading": "Decision-Making",                        "category": "governance",             "risk_weight": "HIGH"},
        {"canonical_title": "Transfer Restrictions",                "raw_heading": "Transfer of Interest",                   "category": "ownership",              "risk_weight": "HIGH"},
        {"canonical_title": "Non-Compete Obligation",               "raw_heading": "Non-Compete",                            "category": "restrictive_covenant",   "risk_weight": "HIGH"},
        {"canonical_title": "Exit Mechanism",                       "raw_heading": "Withdrawal and Exit",                    "category": "exit",                   "risk_weight": "HIGH"},
        {"canonical_title": "Dissolution and Winding Up",           "raw_heading": "Dissolution",                            "category": "termination",            "risk_weight": "HIGH"},
        {"canonical_title": "Jurisdiction",                         "raw_heading": "Governing Law",                          "category": "jurisdiction",           "risk_weight": "MEDIUM"},
    ],
}

# Flatten all clauses for UNKNOWN contract type fallback
ALL_CLAUSES = [c for clauses in CLAUSE_LIBRARY.values() for c in clauses]


# MAIN ENTRY POINT
def run(state: PipelineState) -> PipelineState:
    logger.info(f"[{AGENT_NAME}] Starting → {state.file_name}")
    state.ingestion_status = AgentStatus.RUNNING

    error = _validate(state)
    if error:
        return _fail(state, error)
    try:
        _extract_raw(state)
    except Exception as e:
        return _fail(state, f"Extraction failed: {e}")

    _clean_and_normalise(state)
    _detect_contract_type(state)
    _segment_clauses(state)

    if state.scanned_pages:
        logger.warning(f"[{AGENT_NAME}] Scanned pages: {state.scanned_pages} → NEEDS_OCR")
        state.ingestion_status = AgentStatus.NEEDS_OCR
    else:
        state.ingestion_status = AgentStatus.COMPLETED

    logger.info(
        f"[{AGENT_NAME}] Done | Status={state.ingestion_status.value} | "
        f"Type={state.contract_type} ({state.contract_type_confidence}) | "
        f"Pages={state.page_count} | Clauses={len(state.clause_segments)}"
    )
    return state


# VALIDATE + EXTRACT RAW TEXT
def _validate(state: PipelineState) -> str | None:
    file_bytes = state.file_bytes
    state.file_size_kb = round(len(file_bytes) / 1024, 2)
    if len(file_bytes) / (1024 * 1024) > MAX_SIZE_MB:
        return f"File too large. Max: {MAX_SIZE_MB} MB."
    if not file_bytes.startswith(b"%PDF"):
        return "Not a valid PDF (missing %PDF header)."
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return f"Cannot open PDF: {e}"
    if doc.is_encrypted:
        doc.close()
        return "PDF is password-protected."
    if doc.page_count == 0:
        doc.close()
        return "PDF has no pages."
    doc.close()
    state.doc_hash = hashlib.sha256(file_bytes).hexdigest()
    return None


def _extract_raw(state: PipelineState) -> None:
    doc = fitz.open(stream=state.file_bytes, filetype="pdf")
    state.page_count = doc.page_count
    pages, parts, scanned, warnings = [], [], [], []

    for i in range(doc.page_count):
        raw   = doc[i].get_text("text")
        chars = len(raw.strip())
        is_sc = chars < SCANNED_THRESHOLD
        if is_sc:
            scanned.append(i + 1)
            warnings.append(f"Page {i+1} has {chars} chars — likely scanned.")
        pages.append(PageData(page_number=i+1, text=raw, char_count=chars, is_scanned=is_sc))
        parts.append(raw)

    doc.close()
    state.pages              = pages
    state.scanned_pages      = scanned
    state.ingestion_warnings = warnings
    state.full_text          = "\n\n--- PAGE BREAK ---\n\n".join(parts)


# CLEANING & NORMALISATION
def _clean_and_normalise(state: PipelineState) -> None:
    text = state.full_text
    text = text.replace("--- PAGE BREAK ---", "")
    for old, new in [("\u2019","'"),("\u2018","'"),("\u201c",'"'),
                     ("\u201d",'"'),("\u2013","-"),("\u2014","-")]:
        text = text.replace(old, new)
    text = re.sub(r'(?m)^[\s]*[-]?\s*[Pp]age\s+\d+\s*[-]?\s*$', '', text)
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    state.clean_text = text.strip()


# CONTRACT TYPE DETECTION
def _detect_contract_type(state: PipelineState) -> None:
    text_lower  = state.clean_text.lower()
    fname_lower = state.file_name.lower()
    scores: dict[str, int] = {}

    for contract_type, sigs in CONTRACT_SIGNATURES.items():
        score = 0
        for pattern in sigs["primary"]:
            if re.search(pattern, text_lower):
                score += 10
        for pattern in sigs["secondary"]:
            if re.search(pattern, text_lower):
                score += 2
        scores[contract_type] = score

    logger.info(f"[{AGENT_NAME}] Content scores: {scores}")

    winner       = max(scores, key=scores.get)
    winner_score = scores[winner]

    if winner_score == 0:
        for contract_type, sigs in CONTRACT_SIGNATURES.items():
            for pattern in sigs["primary"]:
                if re.search(pattern, fname_lower):
                    logger.warning(f"[{AGENT_NAME}] No content signal. Filename suggests {contract_type}.")
                    state.contract_type            = contract_type
                    state.contract_type_confidence = "low"
                    state.contract_type_method     = "filename_fallback"
                    return
        state.contract_type            = "UNKNOWN"
        state.contract_type_confidence = "low"
        state.contract_type_method     = "keyword"
        return

    sorted_scores = sorted(scores.values(), reverse=True)
    runner_up     = sorted_scores[1] if len(sorted_scores) > 1 else 0

    if winner_score == runner_up:
        for contract_type, sigs in CONTRACT_SIGNATURES.items():
            for pattern in sigs["primary"]:
                if re.search(pattern, fname_lower) and scores[contract_type] == winner_score:
                    winner = contract_type
                    break

    if winner_score >= 10 and winner_score > (runner_up * 2):
        confidence = "high"
    elif winner_score >= 6:
        confidence = "medium"
    else:
        confidence = "low"

    state.contract_type            = winner
    state.contract_type_confidence = confidence
    state.contract_type_method     = "keyword"
    logger.info(f"[{AGENT_NAME}] Contract type → {winner} (confidence: {confidence})")


# CLAUSE PRESENCE CHECK
def _get_library_for_type(contract_type: str) -> list[dict]:
    """Return clauses for detected type, or all clauses if UNKNOWN."""
    if contract_type in CLAUSE_LIBRARY:
        return CLAUSE_LIBRARY[contract_type]
    return ALL_CLAUSES


def _clause_found_in_text(clause: dict, text_lower: str) -> bool:
    """
    Returns True if this clause appears to exist in the document.
    Checks raw_heading and canonical_title as phrases, then word-level.
    """
    raw = clause["raw_heading"].lower()
    can = clause["canonical_title"].lower()

    # 1. Exact phrase match
    if raw in text_lower or can in text_lower:
        return True

    # 2. All significant words of raw_heading present in text
    raw_words = [w for w in raw.split() if len(w) > 4]
    if raw_words and all(w in text_lower for w in raw_words):
        return True

    return False


def _segment_clauses(state: PipelineState) -> None:
    """
    For every canonical clause in the library for this contract type,
    check if it exists in the document text.
    Populates state.clause_segments with all library clauses,
    each marked found=True or found=False.
    """
    library    = _get_library_for_type(state.contract_type)
    text_lower = state.clean_text.lower()
    segments   = []

    for idx, clause in enumerate(library, start=1):
        found = _clause_found_in_text(clause, text_lower)
        segments.append({
            "id":              idx,
            "canonical_title": clause["canonical_title"],
            "raw_heading":     clause["raw_heading"],
            "category":        clause["category"],
            "risk_weight":     clause["risk_weight"],
            "found":           found,
            "library_matched": found,
        })

    state.clause_segments = segments

    found_count = sum(1 for c in segments if c["found"])
    logger.info(
        f"[{AGENT_NAME}] Clause check: {found_count}/{len(segments)} found "
        f"in {state.contract_type} document"
    )


# HELPER
def _fail(state: PipelineState, error: str) -> PipelineState:
    logger.error(f"[{AGENT_NAME}] FAILED — {error}")
    state.ingestion_status = AgentStatus.FAILED
    state.ingestion_error  = error
    return state