from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import logging
import os
import re
from pathlib import Path
from agent_state import AgentStatus
from groq import Groq
from dotenv import load_dotenv
from config import get_config


"""
LexGuard — Clause Comparison Agent

The agent handles two responsibilities from the workflow:

Clause Embedding + Retrieval Layer
      - Loads the standard clause library for the detected contract type
      - Embeds all standard clauses using sentence-transformers
        (all-MiniLM-L6-v2 — fast, free, runs locally, no API needed)
      - Builds a FAISS index in memory
      - For each FOUND clause from Agent 1, retrieves the best matching
        standard clause using cosine similarity

Clause Comparison Agent
      - For each matched pair (contract clause vs standard clause):
        * Uses similarity score from FAISS as deviation signal
        * Calls Groq LLM to generate a plain-English deviation summary
        * Marks clause as deviated if similarity < threshold
      - Populates state.clause_comparisons

LLM Used   : Llama 3.3 70B via Groq (for deviation summary only)
Embeddings : all-MiniLM-L6-v2 via sentence-transformers (local, free)
Vector DB  : FAISS (in-memory, no server needed)

Input  : PipelineState with clause_segments + contract_type + clean_text
Output : PipelineState with clause_status = COMPLETED / FAILED
"""

logger = logging.getLogger(__name__)

AGENT_NAME           = "ClauseComparisonAgent"
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
LLM_MODEL            = "llama-3.3-70b-versatile"
DEVIATION_THRESHOLD  = 0.75 

# ── Standard clause library 
CLAUSE_LIBRARY = {
    "NDA": [
        {"clause_id": "nda_001", "canonical_title": "Confidential Information Definition",      "raw_heading": "Definition of Confidential Information", "category": "confidentiality", "risk_weight": "HIGH",   "text": "\u201cConfidential Information\u201d means any and all non-public, proprietary, or confidential information disclosed by or on behalf of a Disclosing Party to the Receiving Party, whether orally, visually, electronically, or in writing, including but not limited to trade secrets, business plans, customer data, financial information, and technical data, whether or not marked as confidential, provided that such information is identified as confidential at the time of disclosure or should reasonably be understood to be confidential."},
        {"clause_id": "nda_002", "canonical_title": "Confidentiality Obligations",              "raw_heading": "Obligations of Receiving Party",         "category": "confidentiality", "risk_weight": "HIGH",   "text": "The Receiving Party shall (i) hold all Confidential Information in strict confidence, (ii) not disclose such Confidential Information to any third party except as expressly permitted herein, and (iii) not use such Confidential Information for any purpose other than the Purpose defined in this Agreement."},
        {"clause_id": "nda_003", "canonical_title": "Disclosure to Representatives",            "raw_heading": "Permitted Disclosures",                  "category": "confidentiality", "risk_weight": "HIGH",   "text": "The Receiving Party may disclose Confidential Information to its employees, affiliates, officers, directors, advisors, and agents who have a need to know such information for the Purpose, provided that such persons are bound by confidentiality obligations no less restrictive than those contained herein."},
        {"clause_id": "nda_004", "canonical_title": "Exclusions from Confidential Information", "raw_heading": "Exclusions",                             "category": "confidentiality", "risk_weight": "MEDIUM", "text": "Confidential Information shall not include information that (i) is or becomes publicly available without breach, (ii) is already known to the Receiving Party without restriction, (iii) is independently developed, or (iv) is rightfully obtained from a third party without breach of any obligation."},
        {"clause_id": "nda_005", "canonical_title": "Legal Disclosure",                         "raw_heading": "Compelled Disclosure",                   "category": "legal",           "risk_weight": "HIGH",   "text": "If the Receiving Party is required by law, regulation, or court order to disclose any Confidential Information, it shall, to the extent legally permissible, provide prompt written notice to the Disclosing Party and cooperate in seeking a protective order or other appropriate remedy."},
        {"clause_id": "nda_006", "canonical_title": "Protection Standard",                      "raw_heading": "Standard of Care",                       "category": "security",        "risk_weight": "HIGH",   "text": "The Receiving Party shall protect Confidential Information using at least the same degree of care it uses to protect its own confidential information of a similar nature, but in no event less than a reasonable standard of care."},
        {"clause_id": "nda_007", "canonical_title": "Confidentiality Term",                     "raw_heading": "Term and Survival",                      "category": "term",            "risk_weight": "HIGH",   "text": "This Agreement shall remain in effect for a period of two (2) years from the Effective Date; provided that obligations with respect to Confidential Information shall survive for a period of five (5) years thereafter, or indefinitely with respect to trade secrets."},
        {"clause_id": "nda_008", "canonical_title": "Return of Information",                    "raw_heading": "Return or Destruction",                  "category": "data_handling",   "risk_weight": "HIGH",   "text": "Upon written request or termination of this Agreement, the Receiving Party shall promptly return or destroy all copies of Confidential Information and certify such destruction in writing, except as required for legal or archival purposes."},
        {"clause_id": "nda_009", "canonical_title": "Injunctive Relief",                        "raw_heading": "Remedies",                               "category": "legal_remedy",    "risk_weight": "HIGH",   "text": "The Receiving Party acknowledges that any breach of this Agreement may cause irreparable harm for which monetary damages would be inadequate, and the Disclosing Party shall be entitled to seek injunctive relief in addition to any other remedies available at law or in equity."},
        {"clause_id": "nda_010", "canonical_title": "Jurisdiction",                             "raw_heading": "Governing Law",                          "category": "jurisdiction",    "risk_weight": "MEDIUM", "text": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflict of law principles."},
    ],
    "SLA": [
        {"clause_id": "sla_001", "canonical_title": "Uptime Commitment",      "raw_heading": "Service Availability",          "category": "performance",  "risk_weight": "HIGH",   "text": "The Service Provider shall make the Services available with a Monthly Uptime Percentage of at least 99.9%, excluding Scheduled Maintenance and Excused Downtime."},
        {"clause_id": "sla_002", "canonical_title": "Maintenance Window",     "raw_heading": "Scheduled Maintenance",         "category": "operations",   "risk_weight": "MEDIUM", "text": "Scheduled Maintenance shall be performed during predefined maintenance windows with prior notice of at least forty-eight (48) hours and shall not exceed eight (8) hours per month."},
        {"clause_id": "sla_003", "canonical_title": "Penalty for Downtime",   "raw_heading": "Service Credits",               "category": "penalty",      "risk_weight": "HIGH",   "text": "If the Monthly Uptime Percentage falls below the committed level, the Customer shall be eligible to receive service credits calculated as a percentage of the monthly fees, as set forth in the Service Credit Schedule."},
        {"clause_id": "sla_004", "canonical_title": "Response Time SLA",      "raw_heading": "Incident Response Time",        "category": "support",      "risk_weight": "HIGH",   "text": "The Service Provider shall respond to service incidents within the following timeframes: (i) Critical \u2013 1 hour, (ii) High \u2013 4 hours, (iii) Medium \u2013 8 hours, (iv) Low \u2013 24 hours."},
        {"clause_id": "sla_005", "canonical_title": "Resolution SLA",         "raw_heading": "Resolution Time",               "category": "support",      "risk_weight": "HIGH",   "text": "The Service Provider shall use commercially reasonable efforts to resolve incidents within agreed resolution timeframes based on severity levels."},
        {"clause_id": "sla_006", "canonical_title": "Exclusions from SLA",    "raw_heading": "Excused Downtime",              "category": "exclusions",   "risk_weight": "HIGH",   "text": "Downtime caused by force majeure events, customer misuse, third-party services, or internet failures shall not be considered in calculating uptime."},
        {"clause_id": "sla_007", "canonical_title": "Service Monitoring",     "raw_heading": "Monitoring and Reporting",      "category": "monitoring",   "risk_weight": "MEDIUM", "text": "The Service Provider shall continuously monitor service performance and provide monthly reports detailing uptime, incidents, and SLA compliance metrics."},
        {"clause_id": "sla_008", "canonical_title": "Termination Rights",     "raw_heading": "Termination for Chronic Failure","category": "termination", "risk_weight": "HIGH",   "text": "The Customer may terminate this Agreement without penalty if the Service Provider fails to meet the uptime commitment for three (3) consecutive months or any five (5) months in a rolling twelve-month period."},
        {"clause_id": "sla_009", "canonical_title": "Liability Cap",          "raw_heading": "Limitation of Liability",       "category": "liability",    "risk_weight": "HIGH",   "text": "The total liability of the Service Provider arising out of or related to this Agreement shall not exceed the total fees paid by the Customer in the preceding twelve (12) months."},
        {"clause_id": "sla_010", "canonical_title": "Jurisdiction",           "raw_heading": "Governing Law",                 "category": "jurisdiction", "risk_weight": "MEDIUM", "text": "This Agreement shall be governed by and construed in accordance with the laws of England and Wales."},
    ],
    "VENDOR": [
        {"clause_id": "vendor_001", "canonical_title": "Service Scope",              "raw_heading": "Scope of Services",       "category": "services",        "risk_weight": "HIGH",   "text": "The Vendor shall provide the services described in Statement of Work (\u201cSOW\u201d) attached hereto, in accordance with the timelines, specifications, and service levels set forth therein."},
        {"clause_id": "vendor_002", "canonical_title": "Payment Terms",              "raw_heading": "Fees and Payment Terms",  "category": "payment",         "risk_weight": "HIGH",   "text": "The Client shall pay the fees set forth in the applicable SOW within thirty (30) days of receipt of a valid invoice. Late payments may incur interest at the rate of 1.5% per month or the maximum permitted by law."},
        {"clause_id": "vendor_003", "canonical_title": "Service Level Commitment",  "raw_heading": "Service Levels",          "category": "performance",     "risk_weight": "HIGH",   "text": "The Vendor shall perform the Services in accordance with the service levels specified in the applicable SOW. Failure to meet such service levels may result in service credits or other remedies as defined therein."},
        {"clause_id": "vendor_004", "canonical_title": "Confidentiality Obligations","raw_heading": "Confidentiality",        "category": "confidentiality", "risk_weight": "HIGH",   "text": "Each Party agrees to maintain the confidentiality of the other Party\u2019s Confidential Information and shall not disclose such information except as required to perform its obligations under this Agreement."},
        {"clause_id": "vendor_005", "canonical_title": "IP Ownership",              "raw_heading": "Intellectual Property",   "category": "ip",              "risk_weight": "HIGH",   "text": "All deliverables created by the Vendor under this Agreement shall be the exclusive property of the Client upon full payment, except for any pre-existing materials of the Vendor."},
        {"clause_id": "vendor_006", "canonical_title": "Vendor Indemnity",          "raw_heading": "Indemnification",         "category": "liability",       "risk_weight": "HIGH",   "text": "The Vendor shall indemnify, defend, and hold harmless the Client against any claims, damages, losses, or expenses arising out of (i) breach of this Agreement, (ii) negligence or willful misconduct, or (iii) infringement of intellectual property rights."},
        {"clause_id": "vendor_007", "canonical_title": "Liability Cap",             "raw_heading": "Limitation of Liability", "category": "liability",       "risk_weight": "HIGH",   "text": "Except for liability arising from indemnification obligations, breach of confidentiality, or gross negligence, neither Party\u2019s total liability shall exceed the total fees paid under this Agreement in the preceding twelve (12) months."},
        {"clause_id": "vendor_008", "canonical_title": "Termination Rights",        "raw_heading": "Termination",             "category": "termination",     "risk_weight": "HIGH",   "text": "Either Party may terminate this Agreement for material breach upon thirty (30) days written notice if such breach is not cured within the notice period."},
        {"clause_id": "vendor_009", "canonical_title": "Regulatory Compliance",     "raw_heading": "Compliance with Laws",    "category": "compliance",      "risk_weight": "HIGH",   "text": "The Vendor shall comply with all applicable laws, regulations, and industry standards in the performance of its obligations under this Agreement, including data protection and anti-corruption laws."},
        {"clause_id": "vendor_010", "canonical_title": "Jurisdiction",              "raw_heading": "Governing Law",           "category": "jurisdiction",    "risk_weight": "MEDIUM", "text": "This Agreement shall be governed by and construed in accordance with the laws of the State of California, without regard to conflict of law principles."},
    ],
    "PARTNERSHIP": [
        {"clause_id": "partner_001", "canonical_title": "Formation of Partnership",    "raw_heading": "Formation",             "category": "structure",            "risk_weight": "HIGH",   "text": "The Parties hereby form a partnership under the laws of the specified jurisdiction for the purpose of carrying on the business described herein."},
        {"clause_id": "partner_002", "canonical_title": "Capital Contribution",        "raw_heading": "Capital Contributions", "category": "financial",            "risk_weight": "HIGH",   "text": "Each Partner shall contribute capital in the form and amount specified in Schedule A. No Partner shall be required to make additional contributions unless agreed in writing."},
        {"clause_id": "partner_003", "canonical_title": "Profit Distribution",         "raw_heading": "Profit and Loss Sharing","category": "financial",           "risk_weight": "HIGH",   "text": "Profits and losses of the Partnership shall be allocated among the Partners in proportion to their respective ownership interests, unless otherwise agreed."},
        {"clause_id": "partner_004", "canonical_title": "Governance Structure",        "raw_heading": "Management and Control","category": "governance",           "risk_weight": "HIGH",   "text": "The management of the Partnership shall be vested in the Partners, and decisions shall be made in accordance with voting rights proportional to ownership interests, unless otherwise specified."},
        {"clause_id": "partner_005", "canonical_title": "Reserved Matters",            "raw_heading": "Decision-Making",       "category": "governance",           "risk_weight": "HIGH",   "text": "Certain key decisions, including admission of new partners, dissolution, or sale of substantial assets, shall require unanimous consent of all Partners."},
        {"clause_id": "partner_006", "canonical_title": "Transfer Restrictions",       "raw_heading": "Transfer of Interest",  "category": "ownership",            "risk_weight": "HIGH",   "text": "No Partner may transfer or assign its interest in the Partnership without the prior written consent of the other Partners, except as otherwise permitted herein."},
        {"clause_id": "partner_007", "canonical_title": "Non-Compete Obligation",      "raw_heading": "Non-Compete",           "category": "restrictive_covenant", "risk_weight": "HIGH",   "text": "During the term of this Agreement and for a period of one (1) year thereafter, no Partner shall engage in any business that competes with the Partnership without prior written consent."},
        {"clause_id": "partner_008", "canonical_title": "Exit Mechanism",              "raw_heading": "Withdrawal and Exit",   "category": "exit",                 "risk_weight": "HIGH",   "text": "Any Partner may withdraw from the Partnership upon providing ninety (90) days written notice. The withdrawing Partner\u2019s interest shall be valued and purchased in accordance with Schedule B."},
        {"clause_id": "partner_009", "canonical_title": "Dissolution and Winding Up", "raw_heading": "Dissolution",           "category": "termination",          "risk_weight": "HIGH",   "text": "The Partnership may be dissolved upon mutual agreement of the Partners or upon occurrence of specified events, after which its affairs shall be wound up and assets distributed in accordance with ownership interests."},
        {"clause_id": "partner_010", "canonical_title": "Jurisdiction",               "raw_heading": "Governing Law",         "category": "jurisdiction",         "risk_weight": "MEDIUM", "text": "This Agreement shall be governed by and construed in accordance with the laws of the State of New York."},
    ],
}

ALL_CLAUSES = [c for clauses in CLAUSE_LIBRARY.values() for c in clauses]


# MAIN ENTRY POINT
def run(state):
    logger.info(f"[{AGENT_NAME}] Starting → {len(state.clause_segments)} clauses to compare")
    state.clause_status = AgentStatus.RUNNING

    _load_env()

    # Only process clauses that were FOUND in the document
    found_clauses = [c for c in state.clause_segments if c.get("found")]
    if not found_clauses:
        logger.warning(f"[{AGENT_NAME}] No found clauses to compare.")
        state.clause_status = AgentStatus.COMPLETED
        state.clause_comparisons = []
        return state

    logger.info(f"[{AGENT_NAME}] Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Build FAISS index from standard clause library 
    library = CLAUSE_LIBRARY.get(state.contract_type, ALL_CLAUSES)
    std_texts  = [c["text"] for c in library]
    std_embeddings = embedder.encode(std_texts, convert_to_numpy=True)
    std_embeddings = std_embeddings / np.linalg.norm(std_embeddings, axis=1, keepdims=True)

    dim   = std_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product = cosine similarity on normalised vectors
    index.add(std_embeddings.astype("float32"))
    logger.info(f"[{AGENT_NAME}] FAISS index built: {len(library)} standard clauses")

    # Extract contract clause texts from clean_text 
    contract_clause_texts = _extract_clause_texts(state.clean_text, found_clauses)

    # For each found clause — retrieve + compare
    comparisons = []
    groq_client = _get_groq_client()

    for clause_meta in found_clauses:
        canonical   = clause_meta["canonical_title"]
        category    = clause_meta["category"]
        risk_weight = clause_meta["risk_weight"]

        # Get contract text for this clause
        contract_text = contract_clause_texts.get(canonical, "")
        if not contract_text:
            logger.warning(f"[{AGENT_NAME}] No text extracted for clause: {canonical}")
            continue

        # Embed contract clause
        contract_emb = embedder.encode([contract_text], convert_to_numpy=True)
        contract_emb = contract_emb / np.linalg.norm(contract_emb, axis=1, keepdims=True)

        # Search FAISS for best matching standard clause
        scores, indices = index.search(contract_emb.astype("float32"), k=1)
        similarity      = float(scores[0][0])
        best_match      = library[indices[0][0]]

        is_deviated = similarity < DEVIATION_THRESHOLD

        # Get LLM deviation summary
        deviation_summary = _get_deviation_summary(
            groq_client,
            canonical,
            contract_text,
            best_match["text"],
            similarity,
            is_deviated,
        )

        comparisons.append({
            "clause_id":         best_match["clause_id"],
            "canonical_title":   canonical,
            "category":          category,
            "risk_weight":       risk_weight,
            "contract_text":     contract_text,
            "standard_text":     best_match["text"],
            "similarity_score":  round(similarity, 3),
            "is_deviated":       is_deviated,
            "deviation_summary": deviation_summary,
        })

        logger.info(
            f"[{AGENT_NAME}] {canonical} → similarity={similarity:.3f} "
            f"({'DEVIATED' if is_deviated else 'OK'})"
        )

    state.clause_comparisons = comparisons
    state.clause_status      = AgentStatus.COMPLETED

    deviated = sum(1 for c in comparisons if c["is_deviated"])
    logger.info(
        f"[{AGENT_NAME}] Done — {len(comparisons)} compared, "
        f"{deviated} deviations found"
    )
    return state


# EXTRACT CLAUSE TEXT FROM DOCUMENT
def _extract_clause_texts(clean_text: str, found_clauses: list) -> dict:
    """
    For each found clause, extract the relevant paragraph(s) from the
    document text by searching for the raw_heading near its context.
    Returns dict: { canonical_title → extracted text }
    """
    text_lower = clean_text.lower()
    result     = {}

    for clause in found_clauses:
        canonical = clause["canonical_title"]
        raw       = clause.get("raw_heading", "").lower()

        # Find position of raw_heading in document
        pos = text_lower.find(raw)
        if pos == -1:
            # Try canonical title
            pos = text_lower.find(canonical.lower())
        if pos == -1:
            # Try first 2 significant words
            words = [w for w in raw.split() if len(w) > 4][:2]
            for i, ch in enumerate(text_lower):
                if all(w in text_lower[i:i+200] for w in words):
                    pos = i
                    break

        if pos != -1:
            # Extract ~600 chars after the heading
            snippet = clean_text[pos: pos + 600].strip()
            result[canonical] = snippet
        else:
            result[canonical] = ""

    return result


# LLM DEVIATION SUMMARY
def _get_groq_client():
    """Initialise Groq client. Returns None if not available."""
    try:
        api_key = get_config("GROQ_API_KEY")
        if api_key:
            return Groq(api_key=api_key)
    except ImportError:
        pass
    return None


def _get_deviation_summary(client, clause_name, contract_text, standard_text, similarity, is_deviated):
    """
    Use Groq LLM to generate a plain-English deviation summary.
    Falls back to a rule-based summary if Groq is unavailable.
    """
    if not is_deviated:
        return f"Clause aligns with the standard (similarity: {similarity:.0%}). No significant deviations detected."

    if client is None:
        return f"Deviation detected (similarity: {similarity:.0%}). LLM summary unavailable — GROQ_API_KEY not set."

    prompt = f"""You are a legal contract reviewer.

Compare the CONTRACT CLAUSE below against the STANDARD CLAUSE.
Write a 2-3 sentence deviation summary explaining:
1. What is different or missing compared to the standard
2. What legal risk this creates

Be specific and concise. Do not use bullet points.

CLAUSE NAME: {clause_name}
SIMILARITY SCORE: {similarity:.0%}

STANDARD CLAUSE:
{standard_text}

CONTRACT CLAUSE:
{contract_text}

Write the deviation summary:"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Deviation detected (similarity: {similarity:.0%}). Summary generation failed: {e}"


# HELPERS
def _load_env():
    try:
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
    except ImportError:
        pass


def _fail(state, error):
    logger.error(f"[{AGENT_NAME}] FAILED — {error}")
    state.clause_status = AgentStatus.FAILED
    state.clause_error  = error
    return state