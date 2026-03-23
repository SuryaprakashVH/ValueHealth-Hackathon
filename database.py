import logging
import os
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

"""
LexGuard — Database Layer (MongoDB Atlas)

Handles:
  - Connecting to MongoDB Atlas
  - Saving a completed pipeline review
  - Loading past reviews for chatbot context
  - Searching reviews by contract type, party name, risk level

Collections:
  reviews      — one document per contract review
  chat_sessions — chat history per review session
"""

logger = logging.getLogger(__name__)

DB_NAME              = "lexguard"
COLLECTION_REVIEWS   = "reviews"
COLLECTION_CHAT      = "chat_sessions"

# CONNECTION
def get_db():
    """
    Returns MongoDB database handle.
    Returns None if MONGODB_URI is not set or connection fails.
    """
    _load_env()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        logger.warning("[DB] MONGODB_URI not set — database features disabled.")
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")          # Test connection
        logger.info("[DB] Connected to MongoDB Atlas.")
        return client[DB_NAME]
    except Exception as e:
        logger.error(f"[DB] Connection failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# SAVE REVIEW
# ══════════════════════════════════════════════════════════════════════════

def save_review(state) -> str | None:
    """
    Save a completed pipeline review to MongoDB.
    Returns the inserted document _id as string, or None on failure.
    Skips pdf_bytes — too large for MongoDB document (16MB limit).
    """
    db = get_db()
    if db is None:
        return None

    reg = state.risk_register
    doc = {
        "doc_hash":          state.doc_hash,
        "file_name":         state.file_name,
        "contract_type":     state.contract_type,
        "confidence":        state.contract_type_confidence,
        "uploaded_at":       datetime.utcnow(),
        "page_count":        state.page_count,
        "file_size_kb":      state.file_size_kb,

        # Metadata (parties, dates, terms)
        "contract_metadata": {
            k: v for k, v in state.contract_metadata.items()
            if not k.startswith("_")
        },

        # Risk summary (counts only — fast to query)
        "risk_summary": {
            "high":     sum(1 for r in reg if r["severity"] == "HIGH"),
            "medium":   sum(1 for r in reg if r["severity"] == "MEDIUM"),
            "low":      sum(1 for r in reg if r["severity"] == "LOW"),
            "accepted": sum(1 for r in reg if r["severity"] == "ACCEPTED"),
        },

        # Full risk register
        "risk_register": reg,

        # Clause comparisons (similarity scores + deviations)
        "clause_comparisons": [
            {
                "canonical_title":   c["canonical_title"],
                "category":          c["category"],
                "similarity_score":  c["similarity_score"],
                "is_deviated":       c["is_deviated"],
                "deviation_summary": c["deviation_summary"],
            }
            for c in state.clause_comparisons
        ],

        # Clause presence check
        "clause_segments": [
            {
                "canonical_title": c["canonical_title"],
                "category":        c["category"],
                "found":           c.get("found", False),
                "risk_weight":     c.get("risk_weight", ""),
            }
            for c in state.clause_segments
        ],
    }

    try:
        # Upsert — if same doc_hash exists, update it
        result = db[COLLECTION_REVIEWS].update_one(
            {"doc_hash": state.doc_hash},
            {"$set": doc},
            upsert=True,
        )
        _id = str(result.upserted_id or state.doc_hash)
        logger.info(f"[DB] Review saved — doc_hash: {state.doc_hash[:16]}...")
        return _id
    except Exception as e:
        logger.error(f"[DB] Save failed: {e}")
        return None


# LOAD / SEARCH REVIEWS
def get_review_by_hash(doc_hash: str) -> dict | None:
    """Load a specific review by document hash."""
    db = get_db()
    if db is None:
        return None
    try:
        return db[COLLECTION_REVIEWS].find_one({"doc_hash": doc_hash}, {"_id": 0})
    except Exception as e:
        logger.error(f"[DB] Load failed: {e}")
        return None


def search_reviews(query: dict, limit: int = 10) -> list[dict]:
    """
    Search reviews with a MongoDB query dict.
    Examples:
      search_reviews({"contract_type": "NDA"})
      search_reviews({"risk_summary.high": {"$gt": 0}})
    """
    db = get_db()
    if db is None:
        return []
    try:
        return list(
            db[COLLECTION_REVIEWS]
            .find(query, {"_id": 0, "clause_comparisons": 0})
            .sort("uploaded_at", -1)
            .limit(limit)
        )
    except Exception as e:
        logger.error(f"[DB] Search failed: {e}")
        return []


def get_recent_reviews(limit: int = 5) -> list[dict]:
    """Get the N most recently reviewed contracts."""
    return search_reviews({}, limit=limit)


def get_reviews_by_type(contract_type: str) -> list[dict]:
    """Get all reviews of a specific contract type."""
    return search_reviews({"contract_type": contract_type.upper()})


def get_high_risk_reviews() -> list[dict]:
    """Get all reviews with at least one HIGH risk."""
    return search_reviews({"risk_summary.high": {"$gt": 0}})


def text_search_reviews(keyword: str) -> list[dict]:
    """
    Search reviews where file_name or party names contain the keyword.
    Case-insensitive regex search.
    """
    pattern = re.compile(keyword, re.IGNORECASE)
    return search_reviews({
        "$or": [
            {"file_name":                        pattern},
            {"contract_metadata.parties":        pattern},
            {"contract_metadata.vendor_name":    pattern},
            {"contract_metadata.client_name":    pattern},
            {"contract_metadata.partners":       pattern},
            {"contract_metadata.service_provider": pattern},
        ]
    })


# CHAT HISTORY
def save_message(session_id: str, doc_hash: str, role: str, content: str):
    """Append a message to a chat session."""
    db = get_db()
    if db is None:
        return
    try:
        db[COLLECTION_CHAT].update_one(
            {"session_id": session_id},
            {
                "$set":  {"doc_hash": doc_hash, "updated_at": datetime.utcnow()},
                "$push": {"messages": {
                    "role":      role,
                    "content":   content,
                    "timestamp": datetime.utcnow(),
                }},
                "$setOnInsert": {"created_at": datetime.utcnow()},
            },
            upsert=True,
        )
    except Exception as e:
        logger.error(f"[DB] Chat save failed: {e}")


def get_chat_history(session_id: str) -> list[dict]:
    """Load full chat history for a session."""
    db = get_db()
    if db is None:
        return []
    try:
        doc = db[COLLECTION_CHAT].find_one({"session_id": session_id})
        return doc.get("messages", []) if doc else []
    except Exception as e:
        logger.error(f"[DB] Chat load failed: {e}")
        return []


# HELPER
def _load_env():
    try:
        env_path = Path(__file__).resolve().parent / ".env"
        load_dotenv(dotenv_path=env_path) if env_path.exists() else load_dotenv()
    except ImportError:
        pass