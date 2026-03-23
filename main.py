"""
LexGuard — FastAPI Backend
Run with: uvicorn main:app --reload
or: python main.py
"""

import json
import logging
import io
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn

from orchestrator import run_pipeline
from agent_state import AgentStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LexGuard Backend",
    description="Multi-Agent Contract Review System API",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "LexGuard Backend is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/analyze")
async def analyze_contract(file: UploadFile = File(...), download_report: bool = False):
    """
    Main endpoint to analyze a contract PDF.
    
    Parameters:
    - file: PDF file upload
    - download_report: If True, returns the PDF report as file download. If False, returns JSON with all data.
    
    Returns:
    - JSON with all pipeline results, OR
    - PDF report file (if download_report=True)
    """
    
    # Validate file type
    if file.content_type != "application/pdf" and not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        if not file_bytes:
            raise HTTPException(status_code=400, detail="File is empty")
        
        logger.info(f"[API] Processing file: {file.filename} ({len(file_bytes)} bytes)")
        
        # Run the pipeline
        state = run_pipeline(file_bytes, file.filename)
        
        # If user wants PDF report download
        if download_report:
            if not state.report_pdf_bytes:
                raise HTTPException(
                    status_code=500,
                    detail="Report generation failed"
                )
            
            fname = state.file_name.replace(".pdf", "").replace(" ", "_")
            logger.info(f"[API] Returning PDF report ({len(state.report_pdf_bytes)} bytes)")
            
            return StreamingResponse(
                io.BytesIO(state.report_pdf_bytes),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=LexGuard_RiskBrief_{fname}.pdf"}
            )
        
        # Otherwise return full JSON with all results
        response_data = {
            "file_name": state.file_name,
            "doc_hash": state.doc_hash,
            "ingestion_status": state.ingestion_status.value,
            "contract_type": state.contract_type,
            "contract_type_confidence": state.contract_type_confidence,
            "contract_type_method": state.contract_type_method,
            "page_count": state.page_count,
            "file_size_kb": state.file_size_kb,
            "scanned_pages": state.scanned_pages,
            "ingestion_warnings": state.ingestion_warnings,
            "full_text": state.full_text,
            "clean_text": state.clean_text,
            "metadata_status": state.metadata_status.value,
            "contract_metadata": state.contract_metadata,
            "clause_status": state.clause_status.value,
            "clause_segments": [
                {k: v for k, v in c.items()} for c in state.clause_segments
            ] if state.clause_segments else [],
            "clause_comparisons": state.clause_comparisons or [],
            "risk_status": state.risk_status.value,
            "risk_register": state.risk_register or [],
            "report_status": state.report_status.value,
            "report_pdf_bytes": base64.b64encode(state.report_pdf_bytes).decode("utf-8") if state.report_pdf_bytes else None,
            "pages": [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "char_count": p.char_count,
                    "is_scanned": p.is_scanned,
                }
                for p in state.pages
            ] if state.pages else [],
        }
        
        logger.info(f"[API] Pipeline completed. Status: {state.ingestion_status.value}")
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )


@app.get("/api/status")
async def pipeline_status():
    """Get information about available agent statuses"""
    return {
        "available_statuses": [status.value for status in AgentStatus],
        "description": "Possible pipeline agent statuses"
    }


if __name__ == "__main__":
    # Run FastAPI server
    logger.info("[Server] Starting LexGuard Backend on http://localhost:8000")
    logger.info("[Server] API docs available at http://localhost:8000/docs")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
