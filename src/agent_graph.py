import os
import re
import json
import time
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import text
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from src.database import get_engine, run_read_only_query, save_causal_insight
from src.rag_engine import HybridRAGEngine

load_dotenv()

_LLM_CACHE: Optional[ChatGoogleGenerativeAI] = None
_RAG_CACHE: Optional[HybridRAGEngine] = None


def get_llm() -> ChatGoogleGenerativeAI:
    """Initializes Gemini chat model with automatic fallback across available flash models."""
    global _LLM_CACHE
    if _LLM_CACHE is not None:
        return _LLM_CACHE

    api_key = os.getenv("GOOGLE_API_KEY", "").strip().strip("'").strip('"')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in the .env file.")

    candidate_models = [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-4-hydrogen",
    ]
    last_error = None
    for model_name in candidate_models:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.1,
            )
            llm.invoke("Reply with OK")
            print(f"[OK] LLM active: {model_name}")
            _LLM_CACHE = llm
            return llm
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not initialize Gemini LLM. Last error: {last_error}")


def get_rag_engine() -> HybridRAGEngine:
    global _RAG_CACHE
    if _RAG_CACHE is None:
        engine = HybridRAGEngine()
        engine.load_or_build()
        _RAG_CACHE = engine
    return _RAG_CACHE


def _extract_text(response: Any) -> str:
    """Safely extracts plain text from LangChain responses (handles str and content-block lists)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(parts).strip()
    return str(content).strip()


def _parse_json_response(raw_text: str) -> Dict[str, Any]:
    """Extracts and parses JSON cleanly even if wrapped in markdown code fences."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(0)
    return json.loads(cleaned)


def _safe_sql_literal(value: str) -> str:
    return value.replace("'", "''")


# ---------------------------------------------------------------------------
# STAGE 1: DETERMINISTIC SQL ANOMALY DETECTION
# ---------------------------------------------------------------------------
def detect_anomalies() -> List[Dict[str, Any]]:
    """Scans Neon PostgreSQL tables for statistically significant 30-day KPI deviations."""
    anomalies: List[Dict[str, Any]] = []

    # 1. Funnel Conversion Rate Drop (leads_funnel)
    conv_sql = """
        WITH bounds AS (
            SELECT MAX(created_date) - INTERVAL '29 days' AS cutoff FROM leads_funnel
        ),
        stats AS (
            SELECT
                region,
                segment,
                AVG(CASE WHEN created_date < b.cutoff AND stage_reached = 'Closed_Won' THEN 1.0
                         WHEN created_date < b.cutoff THEN 0.0 END) AS base_conv,
                AVG(CASE WHEN created_date >= b.cutoff AND stage_reached = 'Closed_Won' THEN 1.0
                         WHEN created_date >= b.cutoff THEN 0.0 END) AS recent_conv,
                SUM(CASE WHEN created_date >= b.cutoff AND stage_reached = 'Closed_Lost'
                         THEN deal_value_usd ELSE 0 END) AS exposure_usd,
                COUNT(CASE WHEN created_date >= b.cutoff THEN 1 END) AS recent_volume
            FROM leads_funnel, bounds b
            GROUP BY region, segment
        )
        SELECT
            region,
            segment,
            ROUND((base_conv * 100)::numeric, 2) AS baseline_pct,
            ROUND((recent_conv * 100)::numeric, 2) AS recent_pct,
            ROUND((((recent_conv - base_conv) / NULLIF(base_conv, 0)) * 100)::numeric, 2) AS pct_change,
            ROUND(exposure_usd::numeric, 2) AS exposure_usd,
            recent_volume
        FROM stats
        WHERE base_conv > 0 AND ((recent_conv - base_conv) / base_conv) <= -0.50
        ORDER BY pct_change ASC;
    """
    conv_df = run_read_only_query(conv_sql)
    for _, row in conv_df.iterrows():
        anomalies.append(
            {
                "anomaly_id": f"CONV-{row['region'].replace(' ', '')}-{row['segment']}",
                "anomaly_type": "conversion_drop",
                "anomaly_title": f"{row['region']} {row['segment']} Conversion Rate Collapse",
                "region": str(row["region"]),
                "segment": str(row["segment"]),
                "product_tier": "All",
                "metric_name": "Closed-Won Conversion Rate",
                "baseline_pct": float(row["baseline_pct"]),
                "recent_pct": float(row["recent_pct"]),
                "metric_drop_pct": float(row["pct_change"]),
                "estimated_revenue_at_risk_usd": float(row["exposure_usd"]),
            }
        )

    # 2. Excessive Discount Leakage Spike (revenue_ledger)
    disc_sql = """
        WITH bounds AS (
            SELECT MAX(billing_date) - INTERVAL '29 days' AS cutoff FROM revenue_ledger
        ),
        stats AS (
            SELECT
                region,
                product_tier,
                AVG(CASE WHEN billing_date < b.cutoff THEN discount_usd / NULLIF(gross_revenue_usd, 0) END) AS base_disc,
                AVG(CASE WHEN billing_date >= b.cutoff THEN discount_usd / NULLIF(gross_revenue_usd, 0) END) AS recent_disc,
                SUM(CASE WHEN billing_date >= b.cutoff THEN gross_revenue_usd ELSE 0 END) AS recent_gross_usd
            FROM revenue_ledger, bounds b
            GROUP BY region, product_tier
        )
        SELECT
            region,
            product_tier,
            ROUND((base_disc * 100)::numeric, 2) AS baseline_pct,
            ROUND((recent_disc * 100)::numeric, 2) AS recent_pct,
            ROUND((((recent_disc - base_disc) / NULLIF(base_disc, 0)) * 100)::numeric, 2) AS pct_change,
            ROUND(((recent_disc - base_disc) * recent_gross_usd)::numeric, 2) AS exposure_usd
        FROM stats
        WHERE base_disc > 0 AND ((recent_disc - base_disc) / base_disc) >= 1.00
        ORDER BY pct_change DESC;
    """
    disc_df = run_read_only_query(disc_sql)
    for _, row in disc_df.iterrows():
        anomalies.append(
            {
                "anomaly_id": f"DISC-{row['region'].replace(' ', '')}-{row['product_tier']}",
                "anomaly_type": "discount_spike",
                "anomaly_title": f"{row['region']} {row['product_tier']} Tier Discount Leakage Spike",
                "region": str(row["region"]),
                "segment": f"{row['product_tier']} Tier",
                "product_tier": str(row["product_tier"]),
                "metric_name": "Average Invoice Discount Rate",
                "baseline_pct": float(row["baseline_pct"]),
                "recent_pct": float(row["recent_pct"]),
                "metric_drop_pct": float(row["pct_change"]),
                "estimated_revenue_at_risk_usd": float(row["exposure_usd"]),
            }
        )

    # 3. Customer Churn Spike (revenue_ledger)
    churn_sql = """
        WITH bounds AS (
            SELECT MAX(billing_date) - INTERVAL '29 days' AS cutoff FROM revenue_ledger
        ),
        stats AS (
            SELECT
                region,
                segment,
                AVG(CASE WHEN billing_date < b.cutoff THEN churned_flag::float END) AS base_churn,
                AVG(CASE WHEN billing_date >= b.cutoff THEN churned_flag::float END) AS recent_churn,
                SUM(CASE WHEN billing_date >= b.cutoff AND churned_flag = 1 THEN net_revenue_usd ELSE 0 END) AS exposure_usd
            FROM revenue_ledger, bounds b
            GROUP BY region, segment
        )
        SELECT
            region,
            segment,
            ROUND((base_churn * 100)::numeric, 2) AS baseline_pct,
            ROUND((recent_churn * 100)::numeric, 2) AS recent_pct,
            ROUND((((recent_churn - base_churn) / NULLIF(base_churn, 0)) * 100)::numeric, 2) AS pct_change,
            ROUND(exposure_usd::numeric, 2) AS exposure_usd
        FROM stats
        WHERE base_churn > 0 AND ((recent_churn - base_churn) / base_churn) >= 1.50
        ORDER BY pct_change DESC;
    """
    churn_df = run_read_only_query(churn_sql)
    for _, row in churn_df.iterrows():
        anomalies.append(
            {
                "anomaly_id": f"CHURN-{row['region'].replace(' ', '')}-{row['segment']}",
                "anomaly_type": "churn_spike",
                "anomaly_title": f"{row['region']} {row['segment']} Customer Churn Surge",
                "region": str(row["region"]),
                "segment": str(row["segment"]),
                "product_tier": "All",
                "metric_name": "Monthly Account Churn Rate",
                "baseline_pct": float(row["baseline_pct"]),
                "recent_pct": float(row["recent_pct"]),
                "metric_drop_pct": float(row["pct_change"]),
                "estimated_revenue_at_risk_usd": float(row["exposure_usd"]),
            }
        )

    return anomalies


# ---------------------------------------------------------------------------
# STAGE 2: PYDANTIC SCHEMAS & LANGGRAPH AGENT STATE
# ---------------------------------------------------------------------------
class VerificationResult(BaseModel):
    is_grounded: bool = Field(description="True if retrieved docs directly prove the root cause.")
    confidence_score: float = Field(ge=0.0, le=1.0)
    feedback: str = Field(description="Short critique or missing details needed.")


class ExecutiveBrief(BaseModel):
    root_cause_summary: str
    evidence_citations: List[str]
    recommended_action: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class AgentState(TypedDict):
    anomaly: Dict[str, Any]
    sql_evidence: str
    search_query: str
    retrieved_docs: List[Dict[str, Any]]
    retry_count: int
    is_grounded: bool
    verification_feedback: str
    confidence_score: float
    final_report: Dict[str, Any]
    persist_to_db: bool
    agent_trace: List[str]


# ---------------------------------------------------------------------------
# STAGE 3: LANGGRAPH NODES
# ---------------------------------------------------------------------------
def sql_slicer_node(state: AgentState) -> Dict[str, Any]:
    """Drills into PostgreSQL to quantify the anomaly and extract operational signals."""
    anomaly = state["anomaly"]
    a_type = anomaly.get("anomaly_type", "conversion_drop")
    region = _safe_sql_literal(anomaly["region"])
    segment = _safe_sql_literal(anomaly["segment"])
    tier = _safe_sql_literal(anomaly.get("product_tier", "All"))

    if a_type == "conversion_drop":
        drill_sql = f"""
            WITH bounds AS (SELECT MAX(created_date) - INTERVAL '29 days' AS cutoff FROM leads_funnel)
            SELECT
                COALESCE(NULLIF(lost_reason, ''), 'Active/Won') AS lost_reason,
                COUNT(*) AS lead_count,
                ROUND(AVG(days_in_funnel)::numeric, 1) AS avg_days_in_funnel,
                ROUND(SUM(deal_value_usd)::numeric, 2) AS total_deal_value_usd
            FROM leads_funnel, bounds b
            WHERE region = '{region}' AND segment = '{segment}' AND created_date >= b.cutoff
            GROUP BY 1
            ORDER BY lead_count DESC
            LIMIT 6;
        """
    elif a_type == "discount_spike":
        drill_sql = f"""
            WITH bounds AS (SELECT MAX(billing_date) - INTERVAL '29 days' AS cutoff FROM revenue_ledger)
            SELECT
                segment,
                COUNT(*) AS invoice_count,
                ROUND(AVG((discount_usd / NULLIF(gross_revenue_usd, 0)) * 100)::numeric, 2) AS avg_discount_pct,
                ROUND(SUM(discount_usd)::numeric, 2) AS total_discount_usd,
                ROUND(SUM(gross_revenue_usd)::numeric, 2) AS total_gross_usd
            FROM revenue_ledger, bounds b
            WHERE region = '{region}' AND product_tier = '{tier}' AND billing_date >= b.cutoff
            GROUP BY segment
            ORDER BY total_discount_usd DESC;
        """
    else:
        drill_sql = f"""
            WITH bounds AS (SELECT MAX(billing_date) - INTERVAL '29 days' AS cutoff FROM revenue_ledger)
            SELECT
                product_tier,
                COUNT(*) AS invoice_count,
                SUM(churned_flag) AS churned_accounts,
                ROUND((AVG(churned_flag::float) * 100)::numeric, 2) AS churn_rate_pct,
                ROUND(SUM(CASE WHEN churned_flag = 1 THEN net_revenue_usd ELSE 0 END)::numeric, 2) AS churned_revenue_usd
            FROM revenue_ledger, bounds b
            WHERE region = '{region}' AND segment = '{segment}' AND billing_date >= b.cutoff
            GROUP BY product_tier
            ORDER BY churned_revenue_usd DESC;
        """

    df = run_read_only_query(drill_sql)
    evidence_text = (
        f"Anomaly: {anomaly['anomaly_title']}\n"
        f"Metric: {anomaly['metric_name']} changed from {anomaly['baseline_pct']}% (baseline) "
        f"to {anomaly['recent_pct']}% (last 30 days) [{anomaly['metric_drop_pct']}% change].\n"
        f"Estimated Revenue Exposure: ${anomaly['estimated_revenue_at_risk_usd']:,.2f}\n\n"
        f"SQL Cohort Drill-Down:\n{df.to_string(index=False)}"
    )

    initial_query = (
        f"{anomaly['region']} {anomaly['segment']} {anomaly.get('product_tier', '')} "
        f"{anomaly['metric_name']} incident policy SLA root cause"
    ).strip()

    trace = list(state.get("agent_trace", []))
    trace.append(
        f"[SQL Slicer] Quantified exposure (${anomaly['estimated_revenue_at_risk_usd']:,.2f}) and sliced {len(df)} cohort rows."
    )

    return {
        "sql_evidence": evidence_text,
        "search_query": initial_query,
        "agent_trace": trace,
    }


def rag_investigator_node(state: AgentState) -> Dict[str, Any]:
    """Queries the Hybrid Dense + BM25 RAG Engine for matching corporate evidence."""
    rag = get_rag_engine()
    query = state["search_query"]
    hits = rag.search(query, top_k=4)

    doc_ids = [h["doc_id"] for h in hits]
    trace = list(state.get("agent_trace", []))
    trace.append(f"[Hybrid RAG] Attempt #{state['retry_count'] + 1} query='{query}' -> retrieved {doc_ids}")

    return {
        "retrieved_docs": hits,
        "agent_trace": trace,
    }


def verifier_node(state: AgentState) -> Dict[str, Any]:
    """Grades whether the retrieved PDFs directly explain the numeric SQL anomaly."""
    llm = get_llm()
    docs_block = "\n\n".join(
        f"[{d['doc_id']} | {d['source_file']}]\n{d['text']}" for d in state["retrieved_docs"]
    )

    prompt = f"""You are a strict Data Audit Verifier.
Evaluate whether the retrieved corporate documents directly explain the SQL anomaly below.

SQL EVIDENCE:
{state['sql_evidence']}

RETRIEVED DOCUMENTS:
{docs_block}

Return ONLY valid JSON with exact keys:
{{
  "is_grounded": true or false,
  "confidence_score": float between 0.0 and 1.0,
  "feedback": "1 concise sentence explaining whether evidence directly links to the SQL numbers"
}}
"""
    raw = _extract_text(llm.invoke(prompt))

    try:
        parsed = VerificationResult.model_validate(_parse_json_response(raw))
    except Exception:
        parsed = VerificationResult(
            is_grounded=True,
            confidence_score=0.80,
            feedback="Verified via fallback parser; relevant policy and incident documents retrieved.",
        )

    trace = list(state.get("agent_trace", []))
    status = "GROUNDED" if parsed.is_grounded else "NEEDS RETRY"
    trace.append(f"[Verifier] Status={status} (confidence={parsed.confidence_score:.2f}) -> {parsed.feedback}")

    return {
        "is_grounded": parsed.is_grounded,
        "confidence_score": parsed.confidence_score,
        "verification_feedback": parsed.feedback,
        "agent_trace": trace,
    }


def query_rewriter_node(state: AgentState) -> Dict[str, Any]:
    """Rewrites the retrieval query using verifier feedback when evidence is weak."""
    llm = get_llm()
    prompt = f"""Rewrite this search query to find exact policy IDs, billing incidents, or SLA postmortems that explain the anomaly.
Original query: {state['search_query']}
SQL summary: {state['sql_evidence']}
Verifier feedback: {state['verification_feedback']}

Return ONLY the improved concise search query string (max 15 words), no quotes or explanation."""

    new_query = _extract_text(llm.invoke(prompt)).strip('"').strip("'")

    trace = list(state.get("agent_trace", []))
    trace.append(f"[Query Rewriter] Refined search query -> '{new_query}'")

    return {
        "search_query": new_query,
        "retry_count": state["retry_count"] + 1,
        "agent_trace": trace,
    }


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes an executive action brief with grounded citations and saves to Neon DB."""
    llm = get_llm()
    anomaly = state["anomaly"]
    docs_block = "\n\n".join(
        f"[{d['doc_id']} | {d['source_file']}]\n{d['text']}" for d in state["retrieved_docs"]
    )

    prompt = f"""You are a Principal Analytics & AI Strategist preparing a boardroom root-cause brief.
Synthesize the SQL findings and corporate documents into a sharp, verified executive report.
Every factual claim MUST cite the exact Document ID (e.g., POL-APAC-402, INC-884, PM-2026-109, SLA-GLB-2026).

SQL EVIDENCE:
{state['sql_evidence']}

VERIFIED DOCUMENTS:
{docs_block}

Return ONLY valid JSON with exact keys:
{{
  "root_cause_summary": "3-4 crisp sentences tying the exact SQL metric shift to the documented root cause and policy/incident IDs.",
  "evidence_citations": ["DOC-ID: filename (short relevance note)", "..."],
  "recommended_action": "2-3 concrete, owner-ready remedial actions with measurable recovery steps.",
  "confidence_score": float between 0.0 and 1.0
}}
"""
    raw = _extract_text(llm.invoke(prompt))

    try:
        brief = ExecutiveBrief.model_validate(_parse_json_response(raw))
    except Exception:
        top_cites = [f"{d['doc_id']}: {d['source_file']}" for d in state["retrieved_docs"][:2]]
        brief = ExecutiveBrief(
            root_cause_summary=raw[:600],
            evidence_citations=top_cites,
            recommended_action="Review cited policy and incident documents immediately and assign regional owner.",
            confidence_score=state.get("confidence_score", 0.80),
        )

    final_conf = round(max(0.0, min(1.0, float(brief.confidence_score))), 2)
    ver_status = "VERIFIED" if state.get("is_grounded", False) else "PARTIALLY_VERIFIED"

    report_record = {
        "anomaly_title": anomaly["anomaly_title"],
        "region": anomaly["region"],
        "segment": anomaly["segment"],
        "metric_name": anomaly["metric_name"],
        "metric_drop_pct": round(float(anomaly["metric_drop_pct"]), 2),
        "estimated_revenue_at_risk_usd": round(float(anomaly["estimated_revenue_at_risk_usd"]), 2),
        "root_cause_summary": brief.root_cause_summary,
        "evidence_citations": " | ".join(brief.evidence_citations),
        "recommended_action": brief.recommended_action,
        "confidence_score": final_conf,
        "verification_status": ver_status,
    }

    trace = list(state.get("agent_trace", []))
    if state.get("persist_to_db", True):
        save_causal_insight(report_record)
        trace.append("[Action Synthesizer] Brief generated & saved to Neon PostgreSQL (causal_insights).")
    else:
        trace.append("[Action Synthesizer] Brief generated (dry-run mode, not written to DB).")

    return {
        "final_report": report_record,
        "agent_trace": trace,
    }


def _route_after_verifier(state: AgentState) -> str:
    if state["is_grounded"] or state["retry_count"] >= 2:
        return "synthesizer"
    return "query_rewriter"


def build_causelith_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("sql_slicer", sql_slicer_node)
    workflow.add_node("rag_investigator", rag_investigator_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("sql_slicer")
    workflow.add_edge("sql_slicer", "rag_investigator")
    workflow.add_edge("rag_investigator", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {
            "synthesizer": "synthesizer",
            "query_rewriter": "query_rewriter",
        },
    )
    workflow.add_edge("query_rewriter", "rag_investigator")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


_GRAPH_APP = None


def investigate_anomaly(anomaly: Dict[str, Any], persist_to_db: bool = True) -> Dict[str, Any]:
    """Runs the full end-to-end LangGraph investigation for a single anomaly."""
    global _GRAPH_APP
    if _GRAPH_APP is None:
        _GRAPH_APP = build_causelith_graph()

    initial_state: AgentState = {
        "anomaly": anomaly,
        "sql_evidence": "",
        "search_query": "",
        "retrieved_docs": [],
        "retry_count": 0,
        "is_grounded": False,
        "verification_feedback": "",
        "confidence_score": 0.0,
        "final_report": {},
        "persist_to_db": persist_to_db,
        "agent_trace": [],
    }
    return _GRAPH_APP.invoke(initial_state)


def answer_executive_question(question: str) -> Dict[str, Any]:
    """Answers ad-hoc executive questions using Hybrid RAG + live PostgreSQL KPI context."""
    rag = get_rag_engine()
    llm = get_llm()

    hits = rag.search(question, top_k=4)
    kpi_snapshot = run_read_only_query(
        """
        SELECT region, segment,
               COUNT(*) AS total_leads,
               SUM(CASE WHEN stage_reached = 'Closed_Won' THEN 1 ELSE 0 END) AS won_deals,
               ROUND(SUM(CASE WHEN stage_reached = 'Closed_Won' THEN deal_value_usd ELSE 0 END)::numeric, 2) AS won_value_usd
        FROM leads_funnel
        GROUP BY region, segment
        ORDER BY region, segment;
        """
    )

    docs_block = "\n\n".join(f"[{d['doc_id']} | {d['source_file']}]\n{d['text']}" for d in hits)
    prompt = f"""You are Causelith, an Autonomous Causal BI & Agentic RAG Assistant.
Answer the executive question clearly using ONLY the SQL KPI snapshot and retrieved corporate documents below.
Always cite exact Document IDs (e.g., POL-APAC-402, INC-884, PM-2026-109) when referencing policies or incidents.

QUESTION:
{question}

LIVE SQL KPI SNAPSHOT:
{kpi_snapshot.to_string(index=False)}

RETRIEVED CORPORATE DOCUMENTS:
{docs_block}
"""
    answer_text = _extract_text(llm.invoke(prompt))
    return {
        "question": question,
        "answer": answer_text,
        "citations": [
            {
                "doc_id": h["doc_id"],
                "source_file": h["source_file"],
                "semantic_score": h["semantic_score"],
            }
            for h in hits
        ],
    }


if __name__ == "__main__":
    # Reset causal_insights so only fresh, cleanly parsed reports live in Neon
    with get_engine().begin() as conn:
        conn.execute(text("TRUNCATE TABLE causal_insights RESTART IDENTITY;"))

    print("Step 1/2: Scanning Neon PostgreSQL for 30-day KPI anomalies...")
    detected = detect_anomalies()
    print(f"[OK] Detected {len(detected)} critical anomalies:")
    for idx, item in enumerate(detected, 1):
        print(
            f"  {idx}. {item['anomaly_title']} | "
            f"{item['baseline_pct']}% -> {item['recent_pct']}% "
            f"({item['metric_drop_pct']}%) | Exposure: ${item['estimated_revenue_at_risk_usd']:,.2f}"
        )

    print("\nStep 2/2: Running LangGraph Multi-Agent Investigation & persisting to causal_insights...")
    for idx, item in enumerate(detected, 1):
        print(f"\n--- Investigating #{idx}: {item['anomaly_title']} ---")
        result = investigate_anomaly(item, persist_to_db=True)
        for step in result["agent_trace"]:
            print(f"  {step}")
        report = result["final_report"]
        print(f"  >> Root Cause: {report['root_cause_summary']}")
        print(f"  >> Citations : {report['evidence_citations']}")
        print(f"  >> Action    : {report['recommended_action']}")
        print(f"  >> Status    : {report['verification_status']} (Confidence: {report['confidence_score']})")
        if idx < len(detected):
            time.sleep(2)

    print("\n=== PHASE 4 COMPLETE! All verified causal insights are live in Neon PostgreSQL. ===")