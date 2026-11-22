import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Sync Streamlit Cloud secrets into environment variables before loading internal modules
try:
    for secret_key in ("DATABASE_URL", "GOOGLE_API_KEY"):
        if secret_key in st.secrets and not os.getenv(secret_key):
            os.environ[secret_key] = str(st.secrets[secret_key])
except Exception:
    pass

from src.database import run_read_only_query
from src.agent_graph import (
    detect_anomalies,
    investigate_anomaly,
    answer_executive_question,
)

# ---------------------------------------------------------------------------
# PAGE CONFIG & CUSTOM DARK-MODE STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Causelith | Autonomous Causal BI & Agentic RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero-box {
        background: linear-gradient(135deg, #111827 0%, #0f172a 55%, #162447 100%);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .hero-title {
        font-size: 2.0rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E5FF, #818CF8, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero-sub {
        color: #94A3B8;
        font-size: 0.98rem;
        margin-bottom: 0;
    }
    .insight-card {
        background: #121826;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 4px solid #00E5FF;
        border-radius: 10px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 1rem;
    }
    .badge-verified {
        display: inline-block;
        background: rgba(16, 185, 129, 0.16);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 999px;
        padding: 0.15rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-risk {
        display: inline-block;
        background: rgba(244, 63, 94, 0.16);
        color: #FB7185;
        border: 1px solid rgba(251, 113, 133, 0.4);
        border-radius: 999px;
        padding: 0.15rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 700;
        margin-left: 0.4rem;
    }
    .trace-step {
        background: #0F172A;
        border-left: 3px solid #818CF8;
        padding: 0.55rem 0.85rem;
        margin-bottom: 0.45rem;
        border-radius: 0 6px 6px 0;
        font-family: monospace;
        font-size: 0.85rem;
        color: #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# CACHED DATA LOADERS
# ---------------------------------------------------------------------------
@st.cache_data(ttl=120)
def load_dashboard_data():
    kpi_leads = run_read_only_query(
        """
        SELECT
            COUNT(*) AS total_leads,
            SUM(CASE WHEN stage_reached = 'Closed_Won' THEN 1 ELSE 0 END) AS won_deals,
            ROUND(AVG(CASE WHEN stage_reached = 'Closed_Won' THEN 100.0 ELSE 0.0 END)::numeric, 2) AS conv_pct,
            ROUND(SUM(CASE WHEN stage_reached = 'Closed_Won' THEN deal_value_usd ELSE 0 END)::numeric, 2) AS won_usd
        FROM leads_funnel;
        """
    ).iloc[0]

    kpi_rev = run_read_only_query(
        """
        SELECT
            COUNT(*) AS total_invoices,
            ROUND(SUM(net_revenue_usd)::numeric, 2) AS net_rev_usd,
            ROUND(SUM(discount_usd)::numeric, 2) AS total_disc_usd,
            ROUND((AVG(churned_flag::float) * 100)::numeric, 2) AS churn_pct
        FROM revenue_ledger;
        """
    ).iloc[0]

    conv_trend = run_read_only_query(
        """
        SELECT
            DATE_TRUNC('month', created_date)::date AS month,
            region,
            ROUND(AVG(CASE WHEN stage_reached = 'Closed_Won' THEN 100.0 ELSE 0.0 END)::numeric, 2) AS conversion_pct
        FROM leads_funnel
        WHERE segment = 'Enterprise'
        GROUP BY 1, 2
        ORDER BY 1, 2;
        """
    )

    disc_trend = run_read_only_query(
        """
        SELECT
            DATE_TRUNC('month', billing_date)::date AS month,
            region,
            ROUND(AVG((discount_usd / NULLIF(gross_revenue_usd, 0)) * 100)::numeric, 2) AS avg_discount_pct
        FROM revenue_ledger
        WHERE product_tier = 'Pro'
        GROUP BY 1, 2
        ORDER BY 1, 2;
        """
    )

    churn_trend = run_read_only_query(
        """
        SELECT
            DATE_TRUNC('month', billing_date)::date AS month,
            region,
            ROUND((AVG(churned_flag::float) * 100)::numeric, 2) AS churn_rate_pct
        FROM revenue_ledger
        WHERE segment = 'Mid-Market'
        GROUP BY 1, 2
        ORDER BY 1, 2;
        """
    )

    insights_df = run_read_only_query(
        """
        SELECT
            insight_id,
            created_at,
            anomaly_title,
            region,
            segment,
            metric_name,
            metric_drop_pct,
            estimated_revenue_at_risk_usd,
            root_cause_summary,
            evidence_citations,
            recommended_action,
            confidence_score,
            verification_status
        FROM causal_insights
        ORDER BY insight_id DESC;
        """
    )

    anomalies = detect_anomalies()
    return kpi_leads, kpi_rev, conv_trend, disc_trend, churn_trend, insights_df, anomalies


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ Causelith Engine")
    st.caption("Autonomous Causal BI & Agentic RAG")
    st.divider()
    st.markdown("**Active Pipeline Stack**")
    st.markdown(
        """
        - **Warehouse:** Neon PostgreSQL (`21,000` rows)
        - **Hybrid RAG:** Gemini Embeddings + BM25 + RRF
        - **Orchestration:** LangGraph (Self-Correcting Loop)
        - **LLM Reasoning:** Gemini 3.5 Flash Lite
        - **API Layer:** FastAPI (`/docs`)
        """
    )
    st.divider()
    if st.button("Refresh Live Neon Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">CAUSELITH — Autonomous Causal BI & Agentic RAG</div>
        <p class="hero-sub">
            Detects statistical KPI anomalies across PostgreSQL, runs multi-agent SQL + Hybrid RAG investigations over corporate PDFs,
            verifies grounding against hallucinations, and writes audited action briefs back to the warehouse.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_leads, kpi_rev, conv_trend, disc_trend, churn_trend, insights_df, anomalies = load_dashboard_data()

total_exposure = sum(a["estimated_revenue_at_risk_usd"] for a in anomalies)

# Top Metric Row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net Revenue (6M)", f"${float(kpi_rev['net_rev_usd']) / 1e6:.2f}M", f"{int(kpi_rev['total_invoices']):,} invoices")
c2.metric("Closed-Won Pipeline", f"${float(kpi_leads['won_usd']) / 1e6:.2f}M", f"{float(kpi_leads['conv_pct'])}% win rate")
c3.metric("Active Anomalies", f"{len(anomalies)}", "30-day scan", delta_color="inverse")
c4.metric("Revenue at Risk", f"${total_exposure / 1e6:.2f}M", "3 root causes", delta_color="inverse")
c5.metric("Verified AI Briefs", f"{len(insights_df)}", "Synced to Neon DB")

st.markdown("")

# ---------------------------------------------------------------------------
# NAVIGATION TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Executive BI Telemetry",
        "🤖 Live Multi-Agent Investigator",
        "📑 Verified Causal Insights Ledger",
        "💬 Executive AI Copilot",
    ]
)

# === TAB 1: EXECUTIVE BI TELEMETRY ===
with tab1:
    st.subheader("Live KPI Anomaly Signals (Neon PostgreSQL)")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_conv = px.line(
            conv_trend,
            x="month",
            y="conversion_pct",
            color="region",
            markers=True,
            title="Enterprise Closed-Won Conversion Rate (%) — APAC Collapse",
            template="plotly_dark",
            color_discrete_sequence=["#00E5FF", "#818CF8", "#34D399", "#F59E0B"],
        )
        fig_conv.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18,24,38,0.65)",
            yaxis_title="Conversion Rate (%)",
            xaxis_title="Month",
            legend_title="Region",
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    with col_b:
        fig_disc = px.line(
            disc_trend,
            x="month",
            y="avg_discount_pct",
            color="region",
            markers=True,
            title="Pro Tier Avg Invoice Discount (%) — Europe Stacking Leak",
            template="plotly_dark",
            color_discrete_sequence=["#00E5FF", "#FB7185", "#34D399", "#F59E0B"],
        )
        fig_disc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18,24,38,0.65)",
            yaxis_title="Avg Discount (%)",
            xaxis_title="Month",
            legend_title="Region",
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_disc, use_container_width=True)

    col_c, col_d = st.columns([3, 2])
    with col_c:
        fig_churn = px.bar(
            churn_trend,
            x="month",
            y="churn_rate_pct",
            color="region",
            barmode="group",
            title="Mid-Market Monthly Account Churn Rate (%) — North America Outage Surge",
            template="plotly_dark",
            color_discrete_sequence=["#38BDF8", "#A78BFA", "#34D399", "#FB7185"],
        )
        fig_churn.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18,24,38,0.65)",
            yaxis_title="Churn Rate (%)",
            xaxis_title="Month",
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_churn, use_container_width=True)

    with col_d:
        if anomalies:
            exp_df = pd.DataFrame(anomalies)
            fig_exp = go.Figure(
                go.Bar(
                    x=exp_df["estimated_revenue_at_risk_usd"],
                    y=exp_df["anomaly_title"],
                    orientation="h",
                    marker=dict(color=["#FB7185", "#F59E0B", "#00E5FF"][: len(exp_df)]),
                    text=[f"${v:,.0f}" for v in exp_df["estimated_revenue_at_risk_usd"]],
                    textposition="auto",
                )
            )
            fig_exp.update_layout(
                title="Estimated Revenue Exposure by Anomaly (USD)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(18,24,38,0.65)",
                xaxis_title="Exposure (USD)",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=45, b=20),
            )
            st.plotly_chart(fig_exp, use_container_width=True)


# === TAB 2: LIVE MULTI-AGENT INVESTIGATOR ===
with tab2:
    st.subheader("Autonomous LangGraph Root-Cause Investigator")
    st.caption(
        "Select any detected KPI anomaly to execute the live multi-agent pipeline: "
        "SQL Slicer -> Hybrid RAG (Dense + BM25 + RRF) -> Grounding Verifier -> Query Rewriter -> Action Synthesizer."
    )

    if not anomalies:
        st.info("No active anomalies found.")
    else:
        options = {
            f"{a['anomaly_title']}  |  {a['baseline_pct']}% -> {a['recent_pct']}%  |  Exposure: ${a['estimated_revenue_at_risk_usd']:,.2f}": a
            for a in anomalies
        }
        selected_label = st.selectbox("Select Active Anomaly to Investigate Live:", list(options.keys()))
        selected_anomaly = options[selected_label]

        col_btn1, col_btn2 = st.columns([2, 5])
        with col_btn1:
            persist_flag = st.checkbox("Save new run to Neon DB", value=False)
        with col_btn2:
            run_clicked = st.button("🚀 Run Live Multi-Agent Investigation", type="primary")

        if run_clicked:
            with st.spinner("Executing LangGraph workflow (SQL Slicer -> Hybrid RAG -> Verifier -> Synthesizer)..."):
                result = investigate_anomaly(selected_anomaly, persist_to_db=persist_flag)
                st.session_state["last_investigation"] = result
                if persist_flag:
                    st.cache_data.clear()

        if "last_investigation" in st.session_state:
            res = st.session_state["last_investigation"]
            rep = res["final_report"]

            st.markdown("### 1. Multi-Agent Reasoning Trace")
            for step in res["agent_trace"]:
                st.markdown(f"<div class='trace-step'>{step}</div>", unsafe_allow_html=True)

            col_left, col_right = st.columns([3, 2])
            with col_left:
                st.markdown("### 2. Synthesized Executive Brief")
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <span class="badge-verified">{rep['verification_status']} | Confidence: {rep['confidence_score']}</span>
                        <span class="badge-risk">Exposure: ${rep['estimated_revenue_at_risk_usd']:,.2f}</span>
                        <h4 style="margin-top:0.7rem; color:#F8FAFC;">{rep['anomaly_title']}</h4>
                        <p><strong>Root Cause:</strong> {rep['root_cause_summary']}</p>
                        <p><strong>Recommended Action:</strong> {rep['recommended_action']}</p>
                        <p style="color:#94A3B8; font-size:0.88rem;"><strong>Grounded Citations:</strong> {rep['evidence_citations']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_right:
                st.markdown("### 3. SQL Cohort Evidence")
                st.code(res["sql_evidence"], language="text")


# === TAB 3: VERIFIED CAUSAL INSIGHTS LEDGER ===
with tab3:
    st.subheader("Audited Causal Insights Persisted in Neon PostgreSQL (`causal_insights`)")
    st.caption("This table powers both this executive ledger and downstream Power BI DirectQuery / Import reports.")

    if insights_df.empty:
        st.warning("No insights saved yet. Run `python -m src.agent_graph` or trigger a live investigation.")
    else:
        for _, row in insights_df.iterrows():
            st.markdown(
                f"""
                <div class="insight-card">
                    <span class="badge-verified">#{row['insight_id']} • {row['verification_status']} • Confidence {row['confidence_score']}</span>
                    <span class="badge-risk">{row['region']} • {row['segment']} • ${float(row['estimated_revenue_at_risk_usd']):,.2f}</span>
                    <h4 style="margin-top:0.65rem; color:#F8FAFC;">{row['anomaly_title']} ({row['metric_name']}: {row['metric_drop_pct']}%)</h4>
                    <p><strong>Root Cause:</strong> {row['root_cause_summary']}</p>
                    <p><strong>Action Plan:</strong> {row['recommended_action']}</p>
                    <p style="color:#94A3B8; font-size:0.86rem;"><strong>Evidence Citations:</strong> {row['evidence_citations']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        csv_bytes = insights_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Export Verified Insights to CSV (Power BI / Excel Ready)",
            data=csv_bytes,
            file_name="causelith_verified_insights.csv",
            mime="text/csv",
        )


# === TAB 4: EXECUTIVE AI COPILOT ===
with tab4:
    st.subheader("Executive AI Copilot (Hybrid RAG + Live SQL Context)")
    st.caption("Ask ad-hoc business questions grounded in live Neon PostgreSQL KPIs and corporate policy/postmortem PDFs.")

    preset_cols = st.columns(3)
    presets = [
        "Why did APAC Enterprise conversion collapse and what policy caused it?",
        "Explain the Europe Pro tier discount leak and which ERP rule failed.",
        "What triggered the North America Mid-Market churn spike in November?",
    ]
    for idx, p_text in enumerate(presets):
        if preset_cols[idx].button(p_text, use_container_width=True):
            st.session_state["copilot_query"] = p_text

    user_q = st.text_input(
        "Ask Causelith a business question:",
        value=st.session_state.get("copilot_query", ""),
        placeholder="e.g., Which SLA clauses are at risk due to recent incidents?",
    )

    if st.button("Ask Causelith", type="primary") and user_q.strip():
        with st.spinner("Retrieving corporate documents & live SQL metrics..."):
            qa = answer_executive_question(user_q.strip())
            st.session_state["last_qa"] = qa

    if "last_qa" in st.session_state:
        qa = st.session_state["last_qa"]
        st.markdown("### Grounded Answer")
        st.markdown(qa["answer"])
        st.markdown("#### Retrieved Source Citations")
        cite_df = pd.DataFrame(qa["citations"])
        st.dataframe(cite_df, use_container_width=True, hide_index=True)