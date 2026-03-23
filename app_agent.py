"""
LexGuard — Streamlit UI (Agent version)
Run with: streamlit run app_agent.py
"""

import json
import uuid
import streamlit as st
from orchestrator import run_pipeline
from agent_state import AgentStatus

st.set_page_config(page_title="LexGuard", page_icon="⚖️", layout="wide")

# ── Session state defaults ────────────────────────────────────────────────
# Initialise ALL session state keys upfront before ANY rendering happens.
# This is critical — sidebar reads from these keys, main page writes to them.

if "agent_statuses" not in st.session_state:
    st.session_state.agent_statuses = {
        "Document Ingestion":  AgentStatus.PENDING,
        "Metadata Extraction": AgentStatus.PENDING,
        "Clause Comparison":   AgentStatus.PENDING,
        "Risk Classification": AgentStatus.PENDING,
        "Report Generation":   AgentStatus.PENDING,
    }

if "pipeline_json" not in st.session_state:
    st.session_state.pipeline_json = None

if "show_json" not in st.session_state:
    st.session_state.show_json = False

if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False

# ── Constants ─────────────────────────────────────────────────────────────
STATUS_ICON = {
    AgentStatus.COMPLETED: "🟢",
    AgentStatus.RUNNING:   "🔵",
    AgentStatus.NEEDS_OCR: "🟡",
    AgentStatus.FAILED:    "🔴",
    AgentStatus.PENDING:   "⚪",
}
STATUS_LABEL = {
    AgentStatus.COMPLETED: "Completed",
    AgentStatus.RUNNING:   "Running...",
    AgentStatus.NEEDS_OCR: "Needs OCR",
    AgentStatus.FAILED:    "Failed",
    AgentStatus.PENDING:   "Pending",
}

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — reads from session_state (always up to date)
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚖️ LexGuard")
    st.caption("Multi-Agent Contract Review System")
    st.divider()

    # ── Pipeline Status ───────────────────────────────────────────────────
    st.markdown("#### 🔁 Pipeline Status")

    for agent_name, status in st.session_state.agent_statuses.items():
        icon  = STATUS_ICON.get(status, "⚪")
        label = STATUS_LABEL.get(status, "Pending")
        st.markdown(
            f"{icon} &nbsp; **{agent_name}**  \n"
            f"<span style='font-size:11px;color:gray;padding-left:24px'>{label}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

    st.caption("Agents run in sequence. Each one unlocks the next.")
    st.divider()

    # ── Pipeline State JSON — only shown after pipeline has run ───────────
    if st.session_state.pipeline_json is not None:
        st.markdown("#### 🗂️ Pipeline State JSON")
        st.caption("Live state passed between agents")

        if st.button("Show / Hide JSON", use_container_width=True):
            st.session_state.show_json = not st.session_state.show_json

        if st.session_state.show_json:
            json_section = st.radio(
                "View section",
                ["Summary", "Clauses", "Pages", "Full JSON"],
                label_visibility="collapsed",
            )
            data = st.session_state.pipeline_json

            if json_section == "Summary":
                st.json({k: v for k, v in data.items()
                         if k not in ("full_text", "pages", "clause_segments")})

            elif json_section == "Clauses":
                st.json([
                    {
                        "id":            c.get("id", ""),
                        "canonical_title": c.get("canonical_title", c.get("heading", "")),
                        "category":      c.get("category", c.get("clause_type", "")),
                        "risk_weight":   c.get("risk_weight", ""),
                        "found":         c.get("found", False),
                    }
                    for c in data.get("clause_segments", [])
                ])

            elif json_section == "Pages":
                st.json([
                    {"page": p["page_number"], "chars": p["char_count"],
                     "scanned": p["is_scanned"],
                     "preview": p["text"][:80] + "..." if len(p["text"]) > 80 else p["text"]}
                    for p in data.get("pages", [])
                ])

            elif json_section == "Full JSON":
                display = dict(data)
                if display.get("full_text"):
                    display["full_text"] = display["full_text"][:300] + "... [truncated]"
                display["clause_segments"] = display.get("clause_segments", [])[:3]
                display["_note"] = "clause_segments truncated to first 3 for display"
                st.json(display)

        st.divider()
        st.download_button(
            "⬇ Download full JSON",
            data=json.dumps(st.session_state.pipeline_json, indent=2),
            file_name="pipeline_state.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json_sidebar",
        )
    else:
        st.caption("Upload a contract to see the pipeline JSON here.")


# ══════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════
st.title("⚖️ LexGuard")
st.caption("Multi-Agent Contract Review System — Document Ingestion Agent")
st.divider()

uploaded_file = st.file_uploader(
    "Upload a contract PDF",
    type=["pdf"],
    help="Text-based PDFs up to 20 MB. No password-protected files.",
)

if not uploaded_file:
    # Reset if user clears the file
    if st.session_state.pipeline_ran:
        st.session_state.pipeline_ran = False
        st.session_state.pipeline_json = None
        st.session_state.agent_statuses = {k: AgentStatus.PENDING for k in st.session_state.agent_statuses}
    st.info("Upload a contract PDF. The Ingestion Agent will validate, extract, and hand off to the pipeline.")
    st.stop()

# ── Run pipeline only once per upload ────────────────────────────────────
# Use file name + size as a cache key so re-renders don't re-run the pipeline
file_key = f"{uploaded_file.name}_{uploaded_file.size}"

if st.session_state.get("last_file_key") != file_key:
    # New file uploaded — run the pipeline
    st.session_state.last_file_key = file_key
    st.session_state.pipeline_ran  = False

if not st.session_state.pipeline_ran:
    # Mark ingestion as running first so sidebar updates
    st.session_state.agent_statuses["Document Ingestion"] = AgentStatus.RUNNING

    file_bytes = uploaded_file.read()

    with st.spinner("Ingestion Agent running..."):
        state = run_pipeline(file_bytes, uploaded_file.name)

    # ── Write ALL results into session_state so sidebar can read them ─────
    st.session_state.agent_statuses["Document Ingestion"]  = state.ingestion_status
    st.session_state.agent_statuses["Metadata Extraction"] = state.metadata_status
    # metadata_error stored in state for banner display
    st.session_state.agent_statuses["Clause Comparison"]   = state.clause_status
    st.session_state.agent_statuses["Risk Classification"] = state.risk_status
    st.session_state.agent_statuses["Report Generation"]   = state.report_status
    st.session_state.agent_statuses["Risk Classification"] = state.risk_status
    st.session_state.agent_statuses["Report Generation"]   = state.report_status
    st.session_state.agent_statuses["Report Generation"]   = state.report_status
    st.session_state.agent_statuses["Clause Comparison"]   = state.clause_status
    st.session_state.agent_statuses["Risk Classification"] = state.risk_status
    st.session_state.agent_statuses["Report Generation"]   = state.report_status

    st.session_state.pipeline_json = {
        "file_name":                state.file_name,
        "doc_hash":                 state.doc_hash,
        "ingestion_status":         state.ingestion_status.value,
        "contract_type":            state.contract_type,
        "contract_type_confidence": state.contract_type_confidence,
        "contract_type_method":     state.contract_type_method,
        "page_count":               state.page_count,
        "file_size_kb":             state.file_size_kb,
        "scanned_pages":            state.scanned_pages,
        "warnings":                 state.ingestion_warnings,
        "clause_segments": [
            {k: v for k, v in c.items()} for c in state.clause_segments
        ],
        "contract_metadata": state.contract_metadata,
        "metadata_status":   state.metadata_status.value,
        "full_text":                state.full_text,
        "pages": [
            {"page_number": p.page_number, "text": p.text,
             "char_count": p.char_count,   "is_scanned": p.is_scanned}
            for p in state.pages
        ],
    }
    st.session_state.pipeline_state = state
    st.session_state.pipeline_ran   = True

    # Rerun so sidebar re-renders with updated session_state values
    st.rerun()

# ── From here, read results from session_state (not re-running pipeline) ──
state  = st.session_state.pipeline_state
output = st.session_state.pipeline_json

# ── Persistent download button at top if report is ready ────────────────────
if state.report_pdf_bytes:
    fname = state.file_name.replace(".pdf", "").replace(" ", "_")
    st.download_button(
        label="⬇ Download Legal Risk Brief (PDF)",
        data=state.report_pdf_bytes,
        file_name=f"LexGuard_RiskBrief_{fname}.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary",
        key="dl_pdf_top",
    )

# ── Agent status banner ───────────────────────────────────────────────────
if state.metadata_status == AgentStatus.FAILED:
    st.warning(f"Metadata Extraction failed: {getattr(state, 'metadata_error', 'Unknown error')} — check your GEMINI_API_KEY.")

if state.ingestion_status == AgentStatus.FAILED:
    st.error(f"Ingestion Agent FAILED: {state.ingestion_error}")
    st.stop()
elif state.ingestion_status == AgentStatus.NEEDS_OCR:
    st.warning(f"Scanned pages detected: {state.scanned_pages}. OCR agent will handle these.")
elif state.ingestion_status == AgentStatus.COMPLETED:
    st.success("Ingestion Agent completed successfully. Ready for next agent.")

# ── Metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Pages",         state.page_count)
col2.metric("File size",     f"{state.file_size_kb} KB")
col3.metric("Total chars",   f"{len(state.full_text):,}")
found_n = sum(1 for c in state.clause_segments if c.get("found"))
col4.metric("Clauses found", f"{found_n}/{len(state.clause_segments)}")
col5.metric("Scanned pages", len(state.scanned_pages))
st.caption(f"Document hash (SHA-256): `{state.doc_hash}`")

# ── Contract type ─────────────────────────────────────────────────────────
CONFIDENCE_COLOR = {"high": "🟢", "medium": "🟡", "low": "🔴"}
conf_icon = CONFIDENCE_COLOR.get(state.contract_type_confidence, "⚪")
st.markdown(
    f"**Contract type detected:** `{state.contract_type}` &nbsp; "
    f"{conf_icon} **{state.contract_type_confidence.upper()}** confidence &nbsp; "
    f"*(method: {state.contract_type_method})*"
)

# ── Warnings ──────────────────────────────────────────────────────────────
if state.ingestion_warnings:
    with st.expander(f"⚠️ {len(state.ingestion_warnings)} warning(s)"):
        for w in state.ingestion_warnings:
            st.warning(w)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📄 Extracted text", "🔍 Clause segments", "🧾 Metadata", "⚖️ Clause Comparison", "🚨 Risk Register", "📋 Report", "💬 Chat"])

with tab1:
    view = st.radio("View", ["Full document", "Page by page"],
                    horizontal=True, label_visibility="collapsed")
    if view == "Full document":
        st.text_area("Full document text",
                     value=state.clean_text or state.full_text,
                     height=500, label_visibility="collapsed")
    else:
        selected = st.selectbox("Select page",
                                [p.page_number for p in state.pages],
                                format_func=lambda x: f"Page {x}")
        page = next(p for p in state.pages if p.page_number == selected)
        if page.is_scanned:
            st.warning("Scanned page — OCR needed for full text.")
        st.text_area("Page text",
                     value=page.text or "(No text detected)",
                     height=400, label_visibility="collapsed")
        st.caption(f"Characters: {page.char_count:,}")

with tab2:
    if not state.clause_segments:
        st.info("No clause data available.")
    else:
        found_count   = sum(1 for c in state.clause_segments if c.get("found"))
        missing_count = len(state.clause_segments) - found_count

        # Summary bar
        col_f, col_m, col_t = st.columns(3)
        col_f.metric("✅ Found",   found_count)
        col_m.metric("❌ Missing", missing_count)
        col_t.metric("📋 Total",  len(state.clause_segments))
        st.caption(f"Checked against **{state.contract_type}** standard clause library")
        st.divider()

        RISK_BADGE  = {"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}
        CAT_ICON    = {
            "confidentiality": "🔵", "liability": "🔴", "termination": "🟠",
            "jurisdiction": "🟣",    "ip": "🟡",         "payment": "🟢",
            "performance": "🔵",     "compliance": "🟤", "governance": "🟤",
            "financial": "🟢",       "security": "🔴",   "term": "⚪",
            "data_handling": "🔵",   "legal": "🟣",      "legal_remedy": "🟣",
            "monitoring": "🔵",      "operations": "⚪", "penalty": "🔴",
            "support": "🟢",         "exclusions": "⚪", "structure": "⚪",
            "ownership": "🟡",       "restrictive_covenant": "🟠",
            "exit": "🟠",            "general": "⚪",
        }

        for clause in state.clause_segments:
            found     = clause.get("found", False)
            canonical = clause["canonical_title"]
            category  = clause["category"]
            risk      = clause.get("risk_weight", "MEDIUM")
            raw       = clause.get("raw_heading", "")
            cat_icon  = CAT_ICON.get(category, "⚪")
            status    = "✅ Found" if found else "❌ Missing"

            # Expander label — canonical title + found/missing
            label = f"{status} — {canonical}  [{category.replace('_', ' ').title()}]"

            with st.expander(label, expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Risk weight:** {RISK_BADGE.get(risk, risk)}")
                c2.markdown(f"**Category:** {cat_icon} {category.replace('_',' ').title()}")
                c3.markdown(f"**Document heading:** `{raw}`")

# ── Tab 3: Metadata ───────────────────────────────────────────────────────
with tab3:
    if state.metadata_status == AgentStatus.PENDING:
        st.info("Metadata Extraction Agent has not run yet.")
    elif state.metadata_status == AgentStatus.FAILED:
        st.error(f"Metadata extraction failed: {getattr(state, 'metadata_error', '')}")
    elif not state.contract_metadata:
        st.warning("No metadata extracted.")
    else:
        meta          = state.contract_metadata
        contract_type = meta.get("_contract_type", state.contract_type)
        model_used    = meta.get("_model", "")
        schema_fields = meta.get("_schema_fields", [])

        st.success(f"Metadata extracted using **{model_used}** for **{contract_type}** contract")
        st.divider()

        FIELD_ICONS = {
            "effective_date":         "📅",
            "parties":                "👥",
            "partners":               "👥",
            "term":                   "⏱️",
            "jurisdiction":           "⚖️",
            "confidentiality_period": "🔒",
            "service_provider":       "🏢",
            "customer":               "👤",
            "service_scope":          "📋",
            "vendor_name":            "🏢",
            "client_name":            "👤",
            "payment_terms":          "💰",
            "business_name":          "🏷️",
            "ownership_split":        "📊",
            "key_obligations":        "📝",
        }

        for field in schema_fields:
            value = meta.get(field, "Not found")
            icon  = FIELD_ICONS.get(field, "📌")
            label = field.replace("_", " ").title()

            col_l, col_r = st.columns([1, 2])
            col_l.markdown(f"**{icon} {label}**")
            if isinstance(value, list):
                col_r.markdown("\n".join(f"- {v}" for v in value))
            elif value == "Not found":
                col_r.markdown("*:gray[Not found in document]*")
            else:
                col_r.markdown(str(value))
            st.divider()

st.divider()

# ── Tab 5: Risk Register ──────────────────────────────────────────────────
with tab5:
    if state.risk_status == AgentStatus.PENDING:
        st.info("Risk Classification Agent has not run yet.")
    elif state.risk_status == AgentStatus.FAILED:
        st.error(f"Risk classification failed: {getattr(state, 'risk_error', '')}")
    elif not state.risk_register:
        st.warning("No risk register available.")
    else:
        reg = state.risk_register

        high_items   = [r for r in reg if r["severity"] == "HIGH"]
        medium_items = [r for r in reg if r["severity"] == "MEDIUM"]
        low_items    = [r for r in reg if r["severity"] == "LOW"]
        accepted     = [r for r in reg if r["severity"] == "ACCEPTED"]

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 High",     len(high_items))
        c2.metric("🟡 Medium",   len(medium_items))
        c3.metric("🟢 Low",      len(low_items))
        c4.metric("✅ Accepted", len(accepted))
        st.divider()

        SEVERITY_COLOR = {
            "HIGH":     "🔴",
            "MEDIUM":   "🟡",
            "LOW":      "🟢",
            "ACCEPTED": "✅",
        }

        # Show in order: HIGH first, then MEDIUM, LOW, ACCEPTED
        for item in high_items + medium_items + low_items + accepted:
            sev    = item["severity"]
            icon   = SEVERITY_COLOR.get(sev, "⚪")
            label  = f"{icon} {sev} — {item['canonical_title']}  [{item['category'].replace('_',' ').title()}]"
            expand = sev in ("HIGH", "MEDIUM")

            with st.expander(label, expanded=expand):
                if sev == "ACCEPTED":
                    st.success("Clause aligns with standard. No action required.")
                else:
                    col_a, col_b = st.columns(2)
                    col_a.markdown(f"**Category:** `{item['category'].replace('_',' ').title()}`")
                    col_b.markdown(f"**Similarity:** {item['similarity_score']:.0%}")
                    st.markdown(f"**Business impact:** {item['business_impact']}")
                    st.warning(f"**Recommendation:** {item['recommendation']}")
                    with st.expander("View clause texts", expanded=False):
                        ca, cb = st.columns(2)
                        ca.markdown("**Contract clause**")
                        ca.text(item["contract_text"][:400])
                        cb.markdown("**Standard clause**")
                        cb.text(item["standard_text"][:400])

st.divider()

# ── Tab 6: Report ─────────────────────────────────────────────────────────
with tab6:
    if state.report_status == AgentStatus.PENDING:
        st.info("Report Generation Agent has not run yet.")
    elif state.report_status == AgentStatus.FAILED:
        st.error(f"Report generation failed: {getattr(state, 'report_error', '')}")
    elif not state.report_pdf_bytes:
        st.warning("Report was not generated.")
    else:
        import base64

        fname      = state.file_name.replace(".pdf", "").replace(" ", "_")
        pdf_b64    = base64.b64encode(state.report_pdf_bytes).decode("utf-8")
        size_kb    = round(len(state.report_pdf_bytes) / 1024, 1)
        size_label = f"{size_kb} KB" if size_kb < 1024 else f"{round(size_kb/1024,2)} MB"

        reg    = state.risk_register
        high   = [r for r in reg if r["severity"] == "HIGH"]
        medium = [r for r in reg if r["severity"] == "MEDIUM"]
        low    = [r for r in reg if r["severity"] == "LOW"]
        acc    = [r for r in reg if r["severity"] == "ACCEPTED"]

        # ── Two-column layout: left = info + download, right = PDF preview ──
        col_left, col_right = st.columns([1, 1.6])

        with col_left:
            st.success("Legal Risk Brief is ready")
            st.divider()

            # File info card
            st.markdown("**📄 Report details**")
            st.markdown(f"- File: `LexGuard_RiskBrief_{fname}.pdf`")
            st.markdown(f"- Size: **{size_label}**")
            st.markdown(f"- Contract: `{state.contract_type}` ({state.contract_type_confidence} confidence)")
            st.markdown(f"- Pages reviewed: {state.page_count}")
            st.divider()

            # Risk summary
            st.markdown("**🚨 Risk summary**")
            m1, m2 = st.columns(2)
            m1.metric("🔴 High",   len(high))
            m2.metric("🟡 Medium", len(medium))
            m3, m4 = st.columns(2)
            m3.metric("🟢 Low",      len(low))
            m4.metric("✅ Accepted", len(acc))
            st.divider()

            # Sections list
            st.markdown("**📋 PDF contains**")
            st.markdown("""
- Cover page — name, type, risk counts
- Executive Summary — AI overview
- Contract Metadata — parties, dates, terms
- Risk Register — color-coded table
- Clause Detail — impact + recommendations
- Audit Trail — hash, timestamp, model
""")
            st.divider()

            # Download button with file size
            st.download_button(
                label=f"⬇  Download PDF  ·  {size_label}",
                data=state.report_pdf_bytes,
                file_name=f"LexGuard_RiskBrief_{fname}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="dl_pdf_tab",
            )

st.divider()

# ── Tab 7: Chat ───────────────────────────────────────────────────────────
with tab7:
    # Initialise session state for chat
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.markdown("#### 💬 LexGuard AI Assistant")
    st.caption(
        f"Ask about **{state.file_name}**, search past reviews, "
        "get redline suggestions, or ask general legal questions."
    )

    # Suggested questions
    st.markdown("**Try asking:**")
    col_q1, col_q2, col_q3 = st.columns(3)
    suggestions = [
        ("What are the HIGH risk clauses?",          "current_contract"),
        ("Suggest a redline for the liability cap",  "redline"),
        ("What does indemnification mean?",          "legal_qa"),
        ("Show past vendor agreements reviewed",     "db_search"),
        ("What is the jurisdiction of this contract?", "current_contract"),
        ("Rewrite the termination clause",           "redline"),
    ]
    for i, (suggestion, _) in enumerate(suggestions):
        col = [col_q1, col_q2, col_q3][i % 3]
        if col.button(suggestion, key=f"sugg_{i}", use_container_width=True):
            st.session_state.chat_messages.append({"role": "user", "content": suggestion})
            with st.spinner("Thinking..."):
                try:
                    import chatbot_agent
                    answer = chatbot_agent.answer(
                        question     = suggestion,
                        state        = state,
                        session_id   = st.session_state.chat_session_id,
                        chat_history = st.session_state.chat_messages[:-1],
                    )
                except Exception as e:
                    answer = f"Error: {e}"
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.rerun()

    st.divider()

    # Chat message history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about this contract..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    import chatbot_agent
                    answer = chatbot_agent.answer(
                        question     = prompt,
                        state        = state,
                        session_id   = st.session_state.chat_session_id,
                        chat_history = st.session_state.chat_messages[:-1],
                    )
                except Exception as e:
                    answer = f"Error: {e}"
            st.markdown(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    # Clear chat button
    if st.session_state.chat_messages:
        if st.button("🗑 Clear chat", key="clear_chat"):
            st.session_state.chat_messages = []
            st.session_state.chat_session_id = str(uuid.uuid4())
            st.rerun()

st.divider()

# ── Build + store pipeline state JSON (feeds sidebar viewer) ─────────────
output = {
    "file_name":                state.file_name,
    "doc_hash":                 state.doc_hash,
    "ingestion_status":         state.ingestion_status.value,
    "contract_type":            state.contract_type,
    "contract_type_confidence": state.contract_type_confidence,
    "contract_type_method":     state.contract_type_method,
    "page_count":               state.page_count,
    "file_size_kb":             state.file_size_kb,
    "scanned_pages":            state.scanned_pages,
    "warnings":                 state.ingestion_warnings,
    "metadata_status":          state.metadata_status.value,
        "clause_status":            state.clause_status.value,
        "clause_comparisons":       state.clause_comparisons,
        "risk_status":              state.risk_status.value,
        "risk_register":            state.risk_register,
        "report_status":            state.report_status.value,
    "contract_metadata":        state.contract_metadata,
    "clause_segments": [
        {k: v for k, v in c.items()} for c in state.clause_segments
    ],
    "full_text": state.full_text,
    "pages": [
        {"page_number": p.page_number, "text": p.text,
         "char_count": p.char_count,   "is_scanned": p.is_scanned}
        for p in state.pages
    ],
}
# ── Tab 4: Clause Comparison ─────────────────────────────────────────────
with tab4:
    if state.clause_status == AgentStatus.PENDING:
        st.info("Clause Comparison Agent has not run yet.")
    elif state.clause_status == AgentStatus.FAILED:
        st.error(f"Clause comparison failed: {getattr(state, 'clause_error', '')}")
    elif not state.clause_comparisons:
        st.warning("No clause comparisons available. Ensure clauses were found in the document.")
    else:
        comps     = state.clause_comparisons
        deviated  = sum(1 for c in comps if c["is_deviated"])
        ok        = len(comps) - deviated

        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Aligned",  ok)
        c2.metric("⚠️ Deviated", deviated)
        c3.metric("📋 Compared", len(comps))
        st.divider()

        RISK_COLOR = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}

        for comp in comps:
            score     = comp["similarity_score"]
            deviated_ = comp["is_deviated"]
            risk      = comp.get("risk_weight", "MEDIUM")
            status    = "⚠️ Deviated" if deviated_ else "✅ Aligned"
            pct       = f"{score:.0%}"
            label     = f"{status} — {comp['canonical_title']}  [{comp['category'].replace('_',' ').title()}]  {RISK_COLOR.get(risk,'')} {risk}"

            with st.expander(label, expanded=deviated_):
                col_s, col_r = st.columns([1, 3])
                col_s.metric("Similarity", pct)
                col_r.markdown(f"**Deviation summary:** {comp['deviation_summary']}")
                st.divider()
                ca, cb = st.columns(2)
                ca.markdown("**📄 Contract clause (from document)**")
                ca.text(comp["contract_text"][:400] + ("..." if len(comp["contract_text"]) > 400 else ""))
                cb.markdown("**📚 Standard clause (from library)**")
                cb.text(comp["standard_text"][:400] + ("..." if len(comp["standard_text"]) > 400 else ""))

st.divider()

st.session_state.pipeline_json = output
st.info("Open the sidebar (top-left ▶) to inspect the live pipeline state JSON.", icon="🗂️")
