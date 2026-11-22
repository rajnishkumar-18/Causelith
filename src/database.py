import os
import re
import io
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
def get_engine():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in the .env file.")
    db_url = DATABASE_URL.strip().strip("'").strip('"')
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url, pool_pre_ping=True, pool_recycle=300)


def init_db():
    """Creates the 3 core tables in Neon PostgreSQL if they do not exist."""
    engine = get_engine()
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS leads_funnel (
            lead_id TEXT PRIMARY KEY,
            created_date DATE NOT NULL,
            region TEXT NOT NULL,
            segment TEXT NOT NULL,
            lead_source TEXT NOT NULL,
            stage_reached TEXT NOT NULL,
            deal_value_usd NUMERIC(12, 2) NOT NULL,
            discount_pct NUMERIC(5, 2) NOT NULL,
            days_in_funnel INTEGER NOT NULL,
            lost_reason TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS revenue_ledger (
            invoice_id TEXT PRIMARY KEY,
            billing_date DATE NOT NULL,
            region TEXT NOT NULL,
            segment TEXT NOT NULL,
            product_tier TEXT NOT NULL,
            gross_revenue_usd NUMERIC(12, 2) NOT NULL,
            discount_usd NUMERIC(12, 2) NOT NULL,
            net_revenue_usd NUMERIC(12, 2) NOT NULL,
            churned_flag INTEGER NOT NULL DEFAULT 0
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS causal_insights (
            insight_id SERIAL PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            anomaly_title TEXT NOT NULL,
            region TEXT NOT NULL,
            segment TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_drop_pct NUMERIC(6, 2),
            estimated_revenue_at_risk_usd NUMERIC(14, 2),
            root_cause_summary TEXT NOT NULL,
            evidence_citations TEXT NOT NULL,
            recommended_action TEXT NOT NULL,
            confidence_score NUMERIC(4, 2),
            verification_status TEXT NOT NULL
        );
        """,
    ]

    with engine.begin() as conn:
        for stmt in ddl_statements:
            conn.execute(text(stmt))
    print("[OK] Neon PostgreSQL connected & tables verified.")


def bulk_load_dataframe(df: pd.DataFrame, table_name: str, truncate_first: bool = True):
    """Bulk-loads thousands of rows into PostgreSQL in seconds using COPY."""
    engine = get_engine()
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            if truncate_first:
                cur.execute(f"TRUNCATE TABLE {table_name};")
            buffer = io.StringIO()
            df.to_csv(buffer, index=False, header=False)
            buffer.seek(0)
            cols = ", ".join(df.columns)
            cur.copy_expert(f"COPY {table_name} ({cols}) FROM STDIN WITH (FORMAT CSV)", buffer)
        raw_conn.commit()
    finally:
        raw_conn.close()


def run_read_only_query(sql: str) -> pd.DataFrame:
    """Safe read-only SQL executor for the AI SQL Agent with mutation and injection guardrails."""
    cleaned = sql.strip().rstrip(";")
    upper_sql = cleaned.upper()

    if not (upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")):
        raise ValueError("Security Guard: Only SELECT or WITH queries are allowed.")

    forbidden = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|COPY|EXECUTE)\b"
    if re.search(forbidden, upper_sql):
        raise ValueError("Security Guard: Destructive SQL keyword detected; query blocked.")

    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(cleaned), conn)


def save_causal_insight(insight: dict):
    """Persists a verified AI root-cause report to PostgreSQL for Power BI and the UI."""
    engine = get_engine()
    insert_sql = text(
        """
        INSERT INTO causal_insights (
            anomaly_title, region, segment, metric_name, metric_drop_pct,
            estimated_revenue_at_risk_usd, root_cause_summary,
            evidence_citations, recommended_action, confidence_score, verification_status
        ) VALUES (
            :anomaly_title, :region, :segment, :metric_name, :metric_drop_pct,
            :estimated_revenue_at_risk_usd, :root_cause_summary,
            :evidence_citations, :recommended_action, :confidence_score, :verification_status
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(insert_sql, insight)


if __name__ == "__main__":
    init_db()