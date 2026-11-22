from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.database import run_read_only_query
from src.agent_graph import (
    detect_anomalies,
    investigate_anomaly,
    answer_executive_question,
)

app = FastAPI(
    title="Causelith API — Autonomous Causal BI & Agentic RAG Engine",
    description="REST API exposing real-time SQL anomaly detection, LangGraph multi-agent root-cause investigations, and Hybrid RAG executive Q&A.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Executive business question to investigate.")


class InvestigateRequest(BaseModel):
    anomaly_id: Optional[str] = Field(
        default=None,
        description="Optional anomaly_id from /api/anomalies. If omitted, investigates the top-ranked anomaly.",
    )
    persist_to_db: bool = Field(default=False, description="Persist the new report into causal_insights.")


@app.get("/")
def root():
    return {
        "service": "Causelith — Autonomous Causal BI & Agentic RAG Engine",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "causelith-api", "version": "1.0.0"}


@app.get("/api/kpis")
def get_kpis():
    try:
        leads_summary = run_read_only_query(
            """
            SELECT
                COUNT(*) AS total_leads,
                SUM(CASE WHEN stage_reached = 'Closed_Won' THEN 1 ELSE 0 END) AS closed_won_deals,
                ROUND(AVG(CASE WHEN stage_reached = 'Closed_Won' THEN 100.0 ELSE 0.0 END)::numeric, 2) AS overall_conversion_pct,
                ROUND(SUM(CASE WHEN stage_reached = 'Closed_Won' THEN deal_value_usd ELSE 0 END)::numeric, 2) AS total_won_pipeline_usd
            FROM leads_funnel;
            """
        ).iloc[0].to_dict()

        rev_summary = run_read_only_query(
            """
            SELECT
                COUNT(*) AS total_invoices,
                ROUND(SUM(net_revenue_usd)::numeric, 2) AS total_net_revenue_usd,
                ROUND(SUM(discount_usd)::numeric, 2) AS total_discounts_usd,
                ROUND((AVG(churned_flag::float) * 100)::numeric, 2) AS overall_churn_pct
            FROM revenue_ledger;
            """
        ).iloc[0].to_dict()

        return {"leads_kpis": leads_summary, "revenue_kpis": rev_summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/anomalies")
def list_anomalies():
    try:
        anomalies = detect_anomalies()
        return {"count": len(anomalies), "anomalies": anomalies}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/insights")
def list_insights():
    try:
        df = run_read_only_query(
            """
            SELECT
                insight_id,
                created_at::text AS created_at,
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
        return {"count": len(df), "insights": df.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/investigate")
def run_investigation(payload: InvestigateRequest):
    try:
        anomalies = detect_anomalies()
        if not anomalies:
            return {"message": "No critical anomalies detected.", "result": None}

        target = anomalies[0]
        if payload.anomaly_id:
            matches = [a for a in anomalies if a["anomaly_id"] == payload.anomaly_id]
            if not matches:
                raise HTTPException(status_code=404, detail=f"Anomaly ID '{payload.anomaly_id}' not found.")
            target = matches[0]

        result = investigate_anomaly(target, persist_to_db=payload.persist_to_db)
        return {
            "anomaly": target,
            "agent_trace": result["agent_trace"],
            "sql_evidence": result["sql_evidence"],
            "final_report": result["final_report"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ask")
def ask_copilot(payload: AskRequest):
    try:
        return answer_executive_question(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))