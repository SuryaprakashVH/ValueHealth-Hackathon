import io
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from agent_state import AgentStatus
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, KeepTogether
    )
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from groq import Groq
from dotenv import load_dotenv
from agent_state import AgentStatus

"""
LexGuard — Report Generation Agent

Role         : Consume all pipeline outputs and produce a professional
               Legal Risk Brief as a downloadable PDF.

Responsibility:
    1. Generate Executive Summary via Groq LLM
       (concise paragraph summarising contract + key risks)
    2. Build the full PDF report using ReportLab containing:
       - Cover page (contract name, type, date, doc hash)
       - Executive Summary
       - Contract Metadata table
       - Risk Register (color-coded table: RED/AMBER/GREEN)
       - Clause-level detail (deviation + recommendation per clause)
       - Audit footer (doc hash, timestamp, model used)
    3. Return PDF as bytes → stored in state.report_pdf_bytes
       (UI reads bytes directly and offers st.download_button)

Why ReportLab?
    Pure Python, no external dependencies, works offline,
    produces professional PDFs. No WeasyPrint/wkhtmltopdf needed.

LLM Used : Llama 3.3 70B via Groq (executive summary only)
Input    : Full PipelineState (all previous agents completed)
Output   : state.report_pdf_bytes, state.report_status = COMPLETED
"""


logger = logging.getLogger(__name__)
AGENT_NAME = "ReportGenerationAgent"
LLM_MODEL  = "llama-3.3-70b-versatile"


# MAIN ENTRY POINT
def run(state):
    logger.info(f"[{AGENT_NAME}] Starting report generation...")
    state.report_status = AgentStatus.RUNNING

    _load_env()

    # Step 1 — Generate executive summary via LLM
    exec_summary = _generate_executive_summary(state)

    # Step 2 — Build PDF in memory
    try:
        pdf_bytes = _build_pdf(state, exec_summary)
    except Exception as e:
        return _fail(state, f"PDF generation failed: {e}")

    state.report_pdf_bytes = pdf_bytes
    state.report_status    = AgentStatus.COMPLETED
    logger.info(f"[{AGENT_NAME}] Done — PDF generated ({len(pdf_bytes):,} bytes)")
    return state


# EXECUTIVE SUMMARY VIA LLM
def _generate_executive_summary(state) -> str:
    """Call Groq to write a 3-4 sentence executive summary."""
    groq_client = _get_groq_client()
    if groq_client is None:
        return _fallback_summary(state)

    # Build a compact context for the LLM
    meta    = state.contract_metadata
    reg     = state.risk_register
    high    = [r for r in reg if r["severity"] == "HIGH"]
    medium  = [r for r in reg if r["severity"] == "MEDIUM"]
    deviated = [r for r in reg if r.get("is_deviated")]

    meta_str = "\n".join(
        f"  {k}: {v}" for k, v in meta.items()
        if not k.startswith("_") and v != "Not found"
    )

    risk_str = "\n".join(
        f"  - {r['canonical_title']} ({r['severity']}): {r['deviation_summary']}"
        for r in (high + medium)[:5]
    )

    prompt = f"""You are a senior legal counsel writing an executive summary for a contract review report.

CONTRACT TYPE  : {state.contract_type}
FILE NAME      : {state.file_name}
PAGES          : {state.page_count}

CONTRACT METADATA:
{meta_str or "  Not extracted"}

KEY DEVIATIONS FOUND ({len(deviated)} total, {len(high)} HIGH risk):
{risk_str or "  No significant deviations found."}

Write a professional 3-4 sentence executive summary for this legal risk brief.
Include: what type of contract was reviewed, how many risk issues were found,
which areas carry the most risk, and the overall recommendation (approve / approve with changes / reject).
Be direct and concise. Do not use bullet points."""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        summary = response.choices[0].message.content.strip()
        logger.info(f"[{AGENT_NAME}] Executive summary generated ({len(summary)} chars)")
        return summary
    except Exception as e:
        logger.warning(f"[{AGENT_NAME}] LLM summary failed: {e} — using fallback")
        return _fallback_summary(state)


def _fallback_summary(state) -> str:
    reg      = state.risk_register
    high     = sum(1 for r in reg if r["severity"] == "HIGH")
    medium   = sum(1 for r in reg if r["severity"] == "MEDIUM")
    deviated = sum(1 for r in reg if r.get("is_deviated"))
    return (
        f"This {state.contract_type} contract ({state.file_name}) was reviewed against the "
        f"LexGuard standard clause library. The review identified {deviated} clause deviation(s), "
        f"of which {high} are rated HIGH risk and {medium} MEDIUM risk. "
        f"Legal team review is recommended before execution."
    )


# BUILD PDF WITH REPORTLAB
def _build_pdf(state, exec_summary: str) -> bytes:
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title=f"LexGuard  — {state.file_name}",
        author="LexGuard Multi-Agent System",
    )

    # ── Color palette ─────────────────────────────────────────────────────
    COL_PRIMARY   = colors.HexColor("#1a1a2e")
    COL_ACCENT    = colors.HexColor("#16213e")
    COL_HIGH      = colors.HexColor("#dc3545")
    COL_MEDIUM    = colors.HexColor("#fd7e14")
    COL_LOW       = colors.HexColor("#28a745")
    COL_ACCEPTED  = colors.HexColor("#6c757d")
    COL_HIGH_BG   = colors.HexColor("#fff5f5")
    COL_MEDIUM_BG = colors.HexColor("#fff8f0")
    COL_LOW_BG    = colors.HexColor("#f0fff4")
    COL_HEADER_BG = colors.HexColor("#1a1a2e")
    COL_LIGHT_BG  = colors.HexColor("#f8f9fa")

    # ── Styles ────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()

    def style(name, **kwargs):
        return ParagraphStyle(name, parent=base["Normal"], **kwargs)

    S = {
        "title":     style("title",     fontSize=24, textColor=COL_PRIMARY,  spaceAfter=6,  fontName="Helvetica-Bold", alignment=TA_CENTER),
        "subtitle":  style("subtitle",  fontSize=12, textColor=COL_ACCENT,   spaceAfter=4,  fontName="Helvetica",      alignment=TA_CENTER),
        "h1":        style("h1",        fontSize=14, textColor=COL_PRIMARY,   spaceBefore=16, spaceAfter=6, fontName="Helvetica-Bold"),
        "h2":        style("h2",        fontSize=11, textColor=COL_ACCENT,    spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold"),
        "body":      style("body",      fontSize=9,  textColor=colors.black,  spaceAfter=4,  leading=14),
        "small":     style("small",     fontSize=8,  textColor=colors.gray,   spaceAfter=2),
        "center":    style("center",    fontSize=9,  textColor=colors.black,  alignment=TA_CENTER),
        "bold":      style("bold",      fontSize=9,  textColor=colors.black,  fontName="Helvetica-Bold"),
        "high":      style("high",      fontSize=9,  textColor=COL_HIGH,      fontName="Helvetica-Bold"),
        "medium":    style("medium",    fontSize=9,  textColor=COL_MEDIUM,    fontName="Helvetica-Bold"),
        "low":       style("low",       fontSize=9,  textColor=COL_LOW,       fontName="Helvetica-Bold"),
        "accepted":  style("accepted",  fontSize=9,  textColor=COL_ACCEPTED),
        "footer":    style("footer",    fontSize=7,  textColor=colors.gray,   alignment=TA_CENTER),
    }

    story = []

    # PAGE 1 — COVER
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("LexGuard", S["title"]))
    story.append(Paragraph(" ", style("sub2", fontSize=16, textColor=COL_ACCENT, alignment=TA_CENTER, spaceAfter=2)))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=COL_PRIMARY))
    story.append(Spacer(1, 0.5*cm))

    fname = state.file_name.replace(".pdf", "")
    story.append(Paragraph(fname, style("fn", fontSize=13, textColor=COL_ACCENT, alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4)))

    story.append(Spacer(1, 0.3*cm))
    cover_data = [
        ["Contract Type",  state.contract_type],
        ["Confidence",     state.contract_type_confidence.upper()],
        ["Pages",          str(state.page_count)],
        ["File Size",      f"{state.file_size_kb} KB"],
        ["Document Hash",  (state.doc_hash or "")[:32] + "..."],
        ["Reviewed On",    datetime.now().strftime("%B %d, %Y %H:%M")],
        ["Generated By",   "LexGuard Multi-Agent AI System"],
    ]
    cover_table = Table(cover_data, colWidths=[5*cm, 11*cm])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (0,-1), COL_LIGHT_BG),
        ("FONTNAME",     (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("TEXTCOLOR",    (0,0), (0,-1), COL_PRIMARY),
        ("ALIGN",        (0,0), (-1,-1), "LEFT"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [colors.white, COL_LIGHT_BG]),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.lightgrey),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ]))
    story.append(cover_table)

    # Risk summary on cover
    reg    = state.risk_register
    n_high = sum(1 for r in reg if r["severity"] == "HIGH")
    n_med  = sum(1 for r in reg if r["severity"] == "MEDIUM")
    n_low  = sum(1 for r in reg if r["severity"] == "LOW")
    n_acc  = sum(1 for r in reg if r["severity"] == "ACCEPTED")

    story.append(Spacer(1, 0.8*cm))
    risk_sum = Table(
        [["HIGH", "MEDIUM", "LOW", "ACCEPTED"],
         [str(n_high), str(n_med), str(n_low), str(n_acc)]],
        colWidths=[4*cm]*4
    )
    risk_sum.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,0), COL_HIGH),
        ("BACKGROUND",    (1,0), (1,0), COL_MEDIUM),
        ("BACKGROUND",    (2,0), (2,0), COL_LOW),
        ("BACKGROUND",    (3,0), (3,0), COL_ACCEPTED),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("TEXTCOLOR",     (0,1), (-1,1), COL_PRIMARY),
        ("FONTNAME",      (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,0), 9),
        ("FONTSIZE",      (0,1), (-1,1), 18),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("BOX",           (0,0), (-1,-1), 0.3, colors.lightgrey),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.lightgrey),
    ]))
    story.append(risk_sum)
    story.append(PageBreak())

    # PAGE 2 — EXECUTIVE SUMMARY + METADATA
    story.append(Paragraph("1. Executive Summary", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COL_PRIMARY, spaceAfter=8))
    story.append(Paragraph(exec_summary, S["body"]))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("2. Contract Metadata", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COL_PRIMARY, spaceAfter=8))

    meta = state.contract_metadata
    schema_fields = meta.get("_schema_fields", [])
    if schema_fields:
        meta_rows = []
        for f in schema_fields:
            val = meta.get(f, "Not found")
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            label = f.replace("_", " ").title()
            meta_rows.append([label, str(val)])

        meta_table = Table(meta_rows, colWidths=[5*cm, 11*cm])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,-1), COL_LIGHT_BG),
            ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("TEXTCOLOR",     (0,0), (0,-1), COL_PRIMARY),
            ("ALIGN",         (0,0), (-1,-1), "LEFT"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [colors.white, COL_LIGHT_BG]),
            ("GRID",          (0,0), (-1,-1), 0.3, colors.lightgrey),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        story.append(meta_table)
    else:
        story.append(Paragraph("Metadata extraction did not complete.", S["small"]))

    story.append(PageBreak())

    # PAGE 3 — RISK REGISTER TABLE
    story.append(Paragraph("3. Risk Register", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COL_PRIMARY, spaceAfter=8))

    if reg:
        header = ["#", "Clause", "Category", "Similarity", "Severity"]
        rows   = [header]
        row_styles = []

        sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "ACCEPTED": 3}
        sorted_reg = sorted(reg, key=lambda r: sev_order.get(r["severity"], 99))

        for i, r in enumerate(sorted_reg, 1):
            sev    = r["severity"]
            rows.append([
                str(i),
                r["canonical_title"][:35],
                r["category"].replace("_"," ").title()[:20],
                f"{r['similarity_score']:.0%}",
                sev,
            ])
            ri = len(rows) - 1
            bg = {
                "HIGH":     COL_HIGH_BG,
                "MEDIUM":   COL_MEDIUM_BG,
                "LOW":      COL_LOW_BG,
                "ACCEPTED": colors.white,
            }.get(sev, colors.white)
            tc = {
                "HIGH":     COL_HIGH,
                "MEDIUM":   COL_MEDIUM,
                "LOW":      COL_LOW,
                "ACCEPTED": COL_ACCEPTED,
            }.get(sev, colors.black)
            row_styles.append(("BACKGROUND", (0, ri), (-1, ri), bg))
            row_styles.append(("TEXTCOLOR",  (4, ri), (4,  ri), tc))
            row_styles.append(("FONTNAME",   (4, ri), (4,  ri), "Helvetica-Bold"))

        reg_table = Table(rows, colWidths=[1*cm, 6.5*cm, 4*cm, 2.5*cm, 2.5*cm])
        reg_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), COL_HEADER_BG),
            ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("ALIGN",         (0,0), (-1,-1), "LEFT"),
            ("ALIGN",         (3,0), (4,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("GRID",          (0,0), (-1,-1), 0.3, colors.lightgrey),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            *row_styles,
        ]))
        story.append(reg_table)
    else:
        story.append(Paragraph("No risk data available.", S["small"]))

    story.append(PageBreak())

    # PAGE 4+ — CLAUSE-LEVEL DETAIL
    story.append(Paragraph("4. Clause-Level Risk Detail", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COL_PRIMARY, spaceAfter=8))

    deviated_items = [r for r in sorted_reg if r.get("is_deviated")]

    if not deviated_items:
        story.append(Paragraph("No deviations detected. All clauses align with the standard library.", S["body"]))
    else:
        for idx, r in enumerate(deviated_items, 1):
            sev     = r["severity"]
            sev_col = {"HIGH": COL_HIGH, "MEDIUM": COL_MEDIUM, "LOW": COL_LOW}.get(sev, colors.gray)
            title   = f"{idx}. {r['canonical_title']}"

            block = []
            block.append(Paragraph(title, S["h2"]))

            # Severity + category pill row
            pill_data = [[
                f"Severity: {sev}",
                f"Category: {r['category'].replace('_',' ').title()}",
                f"Similarity: {r['similarity_score']:.0%}",
            ]]
            pill = Table(pill_data, colWidths=[5.3*cm, 5.3*cm, 5.3*cm])
            pill.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (0,0), sev_col),
                ("BACKGROUND",    (1,0), (2,0), COL_LIGHT_BG),
                ("TEXTCOLOR",     (0,0), (0,0), colors.white),
                ("TEXTCOLOR",     (1,0), (2,0), COL_PRIMARY),
                ("FONTNAME",      (0,0), (-1,-1), "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1), 8),
                ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ("BOX",           (0,0), (-1,-1), 0.3, colors.lightgrey),
                ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.lightgrey),
            ]))
            block.append(pill)
            block.append(Spacer(1, 4))

            # Deviation summary
            block.append(Paragraph(f"<b>Deviation:</b> {r['deviation_summary']}", S["body"]))
            block.append(Paragraph(f"<b>Business Impact:</b> {r['business_impact']}", S["body"]))

            # Recommendation box
            rec_table = Table(
                [[Paragraph(f"Recommendation: {r['recommendation']}", S["bold"])]],
                colWidths=[16*cm]
            )
            rec_table.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), COL_MEDIUM_BG),
                ("LEFTPADDING",   (0,0), (-1,-1), 8),
                ("TOPPADDING",    (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("BOX",           (0,0), (-1,-1), 0.5, COL_MEDIUM),
            ]))
            block.append(rec_table)
            block.append(Spacer(1, 10))

            story.append(KeepTogether(block))

    # LAST PAGE — AUDIT TRAIL
    story.append(PageBreak())
    story.append(Paragraph("5. Audit Trail", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COL_PRIMARY, spaceAfter=8))

    audit_data = [
        ["File Name",        state.file_name],
        ["Document Hash",    state.doc_hash or "N/A"],
        ["Contract Type",    f"{state.contract_type} ({state.contract_type_confidence} confidence)"],
        ["Pages",            str(state.page_count)],
        ["Clauses Checked",  str(len(state.clause_segments))],
        ["Clauses Compared", str(len(state.clause_comparisons))],
        ["Risks Found",      f"{n_high} HIGH, {n_med} MEDIUM, {n_low} LOW, {n_acc} ACCEPTED"],
        ["LLM Model",        state.contract_metadata.get("_model", LLM_MODEL)],
        ["Generated At",     datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")],
        ["System",           "LexGuard Multi-Agent AI System v1.0"],
    ]
    audit_table = Table(audit_data, colWidths=[5*cm, 11*cm])
    audit_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), COL_LIGHT_BG),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("TEXTCOLOR",     (0,0), (0,-1), COL_PRIMARY),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [colors.white, COL_LIGHT_BG]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.lightgrey),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("ALIGN",         (0,0), (-1,-1), "LEFT"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(audit_table)
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        "This report was generated automatically by the LexGuard AI system. "
        "It is intended to assist legal professionals in contract review and does not "
        "constitute legal advice. All AI-generated findings should be verified by a "
        "qualified legal professional before contract execution.",
        S["small"]
    ))

    doc.build(story)
    return buf.getvalue()


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


def _fail(state, error):
    logger.error(f"[{AGENT_NAME}] FAILED — {error}")
    state.report_status = AgentStatus.FAILED
    state.report_error  = error
    return state