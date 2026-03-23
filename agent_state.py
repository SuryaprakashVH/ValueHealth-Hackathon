"""
LexGuard — Shared Pipeline State
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class AgentStatus(str, Enum):
    PENDING    = "PENDING"
    RUNNING    = "RUNNING"
    COMPLETED  = "COMPLETED"
    NEEDS_OCR  = "NEEDS_OCR"
    FAILED     = "FAILED"


@dataclass
class PageData:
    page_number: int
    text: str
    char_count: int
    is_scanned: bool


@dataclass
class PipelineState:
    """Shared state baton passed between every agent."""

    # Input 
    file_bytes: bytes = field(default_factory=bytes)
    file_name: str = ""

    # Ingestion Agent outputs
    ingestion_status: AgentStatus = AgentStatus.PENDING
    ingestion_error: Optional[str] = None
    doc_hash: Optional[str] = None
    page_count: int = 0
    file_size_kb: float = 0.0
    full_text: str = ""
    clean_text: str = ""
    pages: list[PageData] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)
    ingestion_warnings: list[str] = field(default_factory=list)
    contract_type: str = "UNKNOWN"
    contract_type_confidence: str = "low"
    contract_type_method: str = ""
    clause_segments: list[dict] = field(default_factory=list)

    # Metadata Extraction Agent outputs
    metadata_status: AgentStatus = AgentStatus.PENDING
    metadata_error: Optional[str] = None
    contract_metadata: dict = field(default_factory=dict)


    # Clause Comparison Agent outputs 
    clause_status: AgentStatus = AgentStatus.PENDING
    clause_error: Optional[str] = None
    clause_comparisons: list[dict] = field(default_factory=list)


    # Risk Classification Agent outputs 
    risk_status: AgentStatus = AgentStatus.PENDING
    risk_error: Optional[str] = None
    risk_register: list[dict] = field(default_factory=list)

    # Report Generation Agent outputs
    report_status: AgentStatus = AgentStatus.PENDING
    report_error: Optional[str] = None
    report_pdf_bytes: bytes = field(default_factory=bytes)