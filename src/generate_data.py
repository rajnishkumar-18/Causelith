import os
from datetime import date, timedelta
import numpy as np
import pandas as pd
from fpdf import FPDF
from src.database import init_db, bulk_load_dataframe


PDF_DIR = os.path.join("data", "pdfs")


def generate_leads_funnel(n_rows: int = 12000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    end_date = date(2026, 11, 21)
    start_date = end_date - timedelta(days=179)
    anomaly_start = end_date - timedelta(days=29)

    regions = ["North America", "Europe", "APAC", "LATAM"]
    region_weights = [0.35, 0.25, 0.25, 0.15]
    segments = ["Enterprise", "Mid-Market", "SMB"]
    segment_weights = [0.25, 0.40, 0.35]
    sources = ["Outbound", "Inbound", "Partner", "Organic"]

    base_deals = {"Enterprise": 48000, "Mid-Market": 16000, "SMB": 4500}

    rows = []
    for i in range(1, n_rows + 1):
        offset = int(rng.integers(0, 180))
        c_date = start_date + timedelta(days=offset)
        region = rng.choice(regions, p=region_weights)
        segment = rng.choice(segments, p=segment_weights)
        source = rng.choice(sources, p=[0.35, 0.30, 0.20, 0.15])

        deal_val = round(float(base_deals[segment] * rng.uniform(0.75, 1.35)), 2)
        discount = round(float(rng.uniform(4.0, 12.0)), 2)
        days_in_funnel = int(rng.integers(10, 32))

        # PLANTED LEAK #1: APAC + Enterprise conversion collapses in last 30 days
        if region == "APAC" and segment == "Enterprise" and c_date >= anomaly_start:
            stage = rng.choice(
                ["MQL", "SQL", "Demo", "Proposal", "Closed_Won", "Closed_Lost"],
                p=[0.12, 0.14, 0.16, 0.10, 0.08, 0.40],
            )
            days_in_funnel = int(rng.integers(28, 52))
            if stage == "Closed_Lost":
                lost_reason = rng.choice(
                    ["Pricing Approval Delay", "SLA Discount Conflict", "Competitor"],
                    p=[0.55, 0.35, 0.10],
                )
            else:
                lost_reason = ""
        else:
            stage = rng.choice(
                ["MQL", "SQL", "Demo", "Proposal", "Closed_Won", "Closed_Lost"],
                p=[0.16, 0.18, 0.18, 0.12, 0.24, 0.12],
            )
            if stage == "Closed_Lost":
                lost_reason = rng.choice(["Budget Freeze", "No Decision", "Competitor"])
            else:
                lost_reason = ""

        rows.append(
            {
                "lead_id": f"LD-{100000 + i}",
                "created_date": c_date.isoformat(),
                "region": region,
                "segment": segment,
                "lead_source": source,
                "stage_reached": stage,
                "deal_value_usd": deal_val,
                "discount_pct": discount,
                "days_in_funnel": days_in_funnel,
                "lost_reason": lost_reason,
            }
        )

    return pd.DataFrame(rows)


def generate_revenue_ledger(n_rows: int = 9000) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    end_date = date(2026, 11, 21)
    start_date = end_date - timedelta(days=179)
    anomaly_start = end_date - timedelta(days=29)

    regions = ["North America", "Europe", "APAC", "LATAM"]
    segments = ["Enterprise", "Mid-Market", "SMB"]
    tiers = ["Core", "Pro", "Enterprise_AI"]
    base_rev = {"Core": 3200, "Pro": 9800, "Enterprise_AI": 28000}

    rows = []
    for i in range(1, n_rows + 1):
        offset = int(rng.integers(0, 180))
        b_date = start_date + timedelta(days=offset)
        region = rng.choice(regions, p=[0.35, 0.25, 0.25, 0.15])
        segment = rng.choice(segments, p=[0.25, 0.40, 0.35])
        tier = rng.choice(tiers, p=[0.35, 0.40, 0.25])

        gross = round(float(base_rev[tier] * rng.uniform(0.8, 1.3)), 2)
        disc_rate = float(rng.uniform(0.05, 0.10))
        churned = int(rng.choice([0, 1], p=[0.965, 0.035]))

        # PLANTED LEAK #2: Europe + Pro tier excessive coupon stacking in last 30 days
        if region == "Europe" and tier == "Pro" and b_date >= anomaly_start:
            disc_rate = float(rng.uniform(0.24, 0.31))

        # PLANTED LEAK #3: North America + Mid-Market churn spike in last 30 days
        if region == "North America" and segment == "Mid-Market" and b_date >= anomaly_start:
            churned = int(rng.choice([0, 1], p=[0.84, 0.16]))

        disc_usd = round(gross * disc_rate, 2)
        net_usd = round(gross - disc_usd, 2)

        rows.append(
            {
                "invoice_id": f"INV-{200000 + i}",
                "billing_date": b_date.isoformat(),
                "region": region,
                "segment": segment,
                "product_tier": tier,
                "gross_revenue_usd": gross,
                "discount_usd": disc_usd,
                "net_revenue_usd": net_usd,
                "churned_flag": churned,
            }
        )

    return pd.DataFrame(rows)


def create_pdf(filename: str, title: str, doc_id: str, metadata_line: str, paragraphs: list[str]):
    os.makedirs(PDF_DIR, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, text=title)
    pdf.ln(2)

    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, text=f"Document ID: {doc_id} | {metadata_line}")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    for para in paragraphs:
        pdf.multi_cell(0, 6, text=para)
        pdf.ln(3)

    pdf.output(os.path.join(PDF_DIR, filename))


def generate_corporate_pdfs():
    docs = [
        # --- LEAK #1 DOCUMENTS (APAC Enterprise) ---
        (
            "Q4_2026_Pricing_Addendum_APAC.pdf",
            "Q4 2026 Regional Pricing Authority Addendum - APAC",
            "POL-APAC-402",
            "Effective Date: October 22, 2026 | Scope: APAC Enterprise Sales",
            [
                "POLICY CHANGE SUMMARY: Effective October 22, 2026, Finance Directive POL-APAC-402 reduced the maximum discount approval threshold for APAC Regional Sales Directors from 20% down to 10% on all Enterprise tier proposals.",
                "APPROVAL BOTTLENECK: Any Enterprise deal requiring an 11% to 20% multi-year volume discount must now route to Global CFO Desk in US Headquarters for manual sign-off, introducing an average approval SLA of 18 to 24 business days.",
                "KNOWN CONFLICT: Legal flagged on October 25 that POL-APAC-402 directly conflicts with Standard APAC Enterprise Master SLA Clause 7.3, which guarantees returning and multi-year Enterprise buyers a baseline 15% volume discount.",
            ],
        ),
        (
            "APAC_Sales_Escalation_Log_Nov2026.pdf",
            "APAC Enterprise Sales Operations Escalation Log",
            "ESC-APAC-114",
            "Period: Oct 23 - Nov 20, 2026 | Region: APAC | Segment: Enterprise",
            [
                "ESCALATION SUMMARY: Between October 23 and November 20, 2026, APAC Enterprise account executives logged 41 critical deal stall tickets tied directly to Finance Policy POL-APAC-402.",
                "CUSTOMER IMPACT: Average Enterprise sales cycle jumped from 21 days to 40+ days. Prospects cited Pricing Approval Delay and SLA Discount Conflict as the top reasons for abandoning final proposals and signing with local competitors.",
                "REMEDIAL RECOMMENDATION: Restore Regional Sales Director discount authority to 16% for multi-year Enterprise contracts compliant with SLA Clause 7.3 and fast-track stalled APAC proposals within 48 hours.",
            ],
        ),
        # --- LEAK #2 DOCUMENTS (Europe Pro Discounting) ---
        (
            "EU_Billing_Incident_INC884.pdf",
            "ERP Billing Incident Postmortem: Double Discount Stacking",
            "INC-884",
            "Detected: October 24, 2026 | Region: Europe | Product Tier: Pro",
            [
                "ROOT CAUSE: During the October 21, 2026 CRM-to-ERP billing connector upgrade in the Europe cluster, validation rule VR-19 (Prevent Promo Code Stacking) was inadvertently disabled for the Pro product tier.",
                "FINANCIAL EFFECT: Customers applying the Q4 promotional code EU-AI-LAUNCH (15% off) automatically received an additional 12% annual prepay discount on top, pushing realized Pro-tier discounts in Europe from an 8% baseline to 27%+.",
                "CORRECTIVE ACTION: Engineering must re-enable ERP validation rule VR-19 immediately, cap combined Europe Pro-tier discounts at 15%, and audit all invoices issued after October 22, 2026.",
            ],
        ),
        (
            "Europe_Q4_Promo_Guidelines_2026.pdf",
            "Europe Commercial Campaign Terms - EU-AI-LAUNCH",
            "MKT-EU-209",
            "Effective Date: October 15, 2026 | Region: Europe | Tier: Pro",
            [
                "CAMPAIGN RULES: Promo code EU-AI-LAUNCH grants a flat 15% promotional discount for Pro tier renewals and new subscriptions in Europe during Q4 2026.",
                "NON-STACKING CLAUSE: Section 3.2 explicitly states that EU-AI-LAUNCH cannot be combined with annual prepay rebates or partner referral discounts. Maximum permitted total invoice discount is 15%.",
            ],
        ),
        # --- LEAK #3 DOCUMENTS (North America Mid-Market Churn) ---
        (
            "NA_Support_Postmortem_Oct2026.pdf",
            "Production Incident Postmortem: CRM Webhook Sync Outage",
            "PM-2026-109",
            "Incident Window: Oct 25 - Oct 31, 2026 | Region: North America | Segment: Mid-Market",
            [
                "INCIDENT OVERVIEW: Deployment of API Gateway v4.2 on October 25 introduced an overly strict webhook rate limit (100 req/min) on the us-east-1 cluster, which primarily hosts North America Mid-Market tenants.",
                "CUSTOMER IMPACT: Bidirectional HubSpot and Salesforce CRM sync failed silently for 6 consecutive days across 310+ Mid-Market accounts. Support received 480+ P1 tickets and multiple renewal cancellation notices.",
                "RETENTION RISK: Affected Mid-Market customers invoked Enterprise SLA Clause 4.1 claiming service availability breach, causing monthly churn in North America Mid-Market to jump sharply in November.",
                "ACTION PLAN: Roll back rate limiter to 1,000 req/min on us-east-1, issue proactive 1-month SLA service credits to affected accounts, and launch executive Customer Success outreach to save pending renewals.",
            ],
        ),
        (
            "Global_Enterprise_SLA_Terms_2026.pdf",
            "Global Master Service Level Agreement (SLA) Reference",
            "SLA-GLB-2026",
            "Effective: FY 2026 | Scope: Global All Segments",
            [
                "CLAUSE 4.1 (UPTIME & INTEGRATION GUARANTEE): Core API and native CRM webhook integrations carry a 99.9% monthly uptime commitment. Outages exceeding 24 consecutive hours entitle customers to penalty-free contract exit or 15% service credit.",
                "CLAUSE 7.3 (MULTI-YEAR VOLUME PRICING): Enterprise accounts committing to 24-month or 36-month agreements qualify for a standard 15% volume discount upon regional director verification.",
            ],
        ),
        # --- NOISE / DISTRACTOR DOCUMENTS (For Realistic RAG Testing) ---
        (
            "Annual_Sales_Playbook_2026.pdf",
            "Global B2B Sales Methodology & Discovery Playbook 2026",
            "PLAY-2026-01",
            "Published: January 2026 | Scope: Global Sales Enablement",
            [
                "DISCOVERY FRAMEWORK: All Inbound and Outbound leads should complete MEDDPICC qualification within 5 business days of entering MQL status.",
                "LATAM GROWTH INITIATIVE: LATAM SMB acquisition targets focus on localized Spanish and Portuguese onboarding webinars with standard 6% quarterly discounts.",
            ],
        ),
        (
            "Global_Security_Compliance_Overview.pdf",
            "SOC 2 Type II and ISO 27001 Annual Compliance Brief",
            "SEC-COMP-2026",
            "Published: August 2026 | Scope: InfoSec & Legal",
            [
                "CERTIFICATION STATUS: All regional cloud clusters passed annual SOC 2 Type II penetration testing with zero critical findings. Data encryption at rest uses AES-256 across PostgreSQL and object storage.",
            ],
        ),
        (
            "Product_Roadmap_Q1_2027.pdf",
            "Executive Product Strategy & AI Roadmap Q1 2027",
            "PROD-2027-Q1",
            "Published: November 2026 | Scope: Product Management",
            [
                "UPCOMING RELEASES: Q1 2027 will introduce autonomous multi-currency forecasting for Enterprise_AI tier accounts and native Snowflake zero-copy connectors.",
            ],
        ),
    ]

    for filename, title, doc_id, meta, paras in docs:
        create_pdf(filename, title, doc_id, meta, paras)
    print(f"[OK] Generated {len(docs)} corporate PDFs in '{PDF_DIR}'.")


def main():
    print("Step 1/4: Initializing Neon PostgreSQL tables...")
    init_db()

    print("Step 2/4: Generating and uploading 12,000 leads_funnel rows...")
    leads_df = generate_leads_funnel(12000)
    bulk_load_dataframe(leads_df, "leads_funnel", truncate_first=True)
    print(f"[OK] {len(leads_df):,} rows uploaded to leads_funnel.")

    print("Step 3/4: Generating and uploading 9,000 revenue_ledger rows...")
    rev_df = generate_revenue_ledger(9000)
    bulk_load_dataframe(rev_df, "revenue_ledger", truncate_first=True)
    print(f"[OK] {len(rev_df):,} rows uploaded to revenue_ledger.")

    print("Step 4/4: Generating corporate PDFs...")
    generate_corporate_pdfs()

    print("\n=== PHASE 2 COMPLETE! Database & PDFs are ready. ===")


if __name__ == "__main__":
    main()