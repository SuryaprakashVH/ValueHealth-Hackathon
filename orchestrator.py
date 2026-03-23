import logging
from agent_state import PipelineState, AgentStatus
from agents import document_ingestion_agent, metadata_extraction_agent, clause_comparison_agent, risk_classification_agent, report_generation_agent
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
import sys
import database

"""
LexGuard — Orchestrator (Mini LangGraph-style pipeline)

This shows EXACTLY how the Document Ingestion Agent fits into the
multi-agent pipeline. The Orchestrator:
  1. Creates the shared PipelineState
  2. Calls each agent in order
  3. Checks each agent's status before proceeding
  4. Routes differently if a special status is detected (e.g. NEEDS_OCR)

In production you'd replace this with LangGraph's StateGraph.
This version runs without installing LangGraph so you can test immediately.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(file_bytes: bytes, file_name: str) -> PipelineState:
    """
    Master pipeline. Runs all agents in sequence.
    Each agent updates the shared state and returns it.
    """

    # Initialise shared state 
    state = PipelineState(file_bytes=file_bytes, file_name=file_name)
    logger.info(f"[Orchestrator] Pipeline started for: {file_name}")

    #  Node 1: Document Ingestion Agent 
    state = document_ingestion_agent.run(state)

    if state.ingestion_status == AgentStatus.FAILED:
        logger.error(f"[Orchestrator] Pipeline aborted at ingestion. Error: {state.ingestion_error}")
        return state

    if state.ingestion_status == AgentStatus.NEEDS_OCR:
        logger.warning(
            f"[Orchestrator] Scanned pages found: {state.scanned_pages}. "
            "Routing to OCR node... (stub — not built yet)"
        )

    #  Node 2: Metadata Extraction Agent 
    state = metadata_extraction_agent.run(state)
    if state.metadata_status == AgentStatus.FAILED:
        logger.error(f"[Orchestrator] Metadata extraction failed: {state.metadata_error}")
    else:
        logger.info("[Orchestrator] Metadata extraction completed.")

    #  Node 3: Clause Comparison Agent
    state = clause_comparison_agent.run(state)
    if state.clause_status == AgentStatus.FAILED:
        logger.error(f"[Orchestrator] Clause comparison failed: {state.clause_error}")
    else:
        deviated = sum(1 for c in state.clause_comparisons if c["is_deviated"])
        logger.info(f"[Orchestrator] Clause comparison done — {deviated} deviations found.")

    # Node 4: Risk Classification Agent 
    state = risk_classification_agent.run(state)
    if state.risk_status == AgentStatus.FAILED:
        logger.error(f"[Orchestrator] Risk classification failed: {state.risk_error}")
    else:
        high = sum(1 for r in state.risk_register if r["severity"] == "HIGH")
        logger.info(f"[Orchestrator] Risk classification done — {high} HIGH risks found.")

    # Node 5: Report Generation Agent 
    state = report_generation_agent.run(state)
    if state.report_status == AgentStatus.FAILED:
        logger.error(f"[Orchestrator] Report generation failed: {state.report_error}")
    else:
        logger.info(f"[Orchestrator] Report generated ({len(state.report_pdf_bytes):,} bytes).")

    #  Save to MongoDB
    db_id = database.save_review(state)
    if db_id:
        logger.info(f"[Orchestrator] Review saved to MongoDB.")
    else:
        logger.warning("[Orchestrator] MongoDB save skipped (DB unavailable or not configured).")

    logger.info(f"[Orchestrator] Pipeline complete. Ingestion status: {state.ingestion_status.value}")
    return state

#  Quick test run
if __name__ == "__main__":
    # Try to create a test PDF
    try:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        c.setFont("Helvetica", 11)
        y = 780
        lines = [
            "NON-DISCLOSURE AGREEMENT",
            "",
            "This Agreement is entered into as of January 1, 2025,",
            "by and between Acme Corp and Beta Ltd.",
            "",
            "1. CONFIDENTIALITY",
            "The Receiving Party shall hold all Confidential Information in strict confidence.",
            "",
            "2. LIABILITY CAP",
            "Total liability shall not exceed USD 50,000 under any circumstances.",
            "",
            "3. TERM",
            "This Agreement shall remain in effect for two (2) years.",
            "",
            "4. GOVERNING LAW",
            "This Agreement is governed by the laws of the State of California.",
        ]
        for line in lines:
            c.drawString(50, y, line)
            y -= 20
        c.save()
        buf.seek(0)
        test_bytes = buf.read()
        test_name  = "test_nda.pdf"
        print("Test PDF created with reportlab.\n")

    except ImportError:
        print("reportlab not installed. Please put a PDF named 'sample.pdf' in this folder.")
        sys.exit(1)

    # Run the pipeline
    final_state = run_pipeline(test_bytes, test_name)

    # Print results
    print("\n" + "="*55)
    print("PIPELINE STATE AFTER INGESTION AGENT")
    print("="*55)
    print(f"  File name     : {final_state.file_name}")
    print(f"  Status        : {final_state.ingestion_status.value}")
    print(f"  Doc hash      : {final_state.doc_hash[:20]}..." if final_state.doc_hash else "  Doc hash      : None")
    print(f"  Pages         : {final_state.page_count}")
    print(f"  File size KB  : {final_state.file_size_kb}")
    print(f"  Total chars   : {len(final_state.full_text):,}")
    print(f"  Scanned pages : {final_state.scanned_pages}")
    print(f"  Warnings      : {len(final_state.ingestion_warnings)}")
    if final_state.ingestion_error:
        print(f"  Error         : {final_state.ingestion_error}")
    print()
    print("TEXT PREVIEW (first 300 chars):")
    print(final_state.full_text[:300])
    print("="*55)