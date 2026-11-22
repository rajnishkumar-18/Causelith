# ⚡ Causelith — Autonomous Causal BI & Agentic RAG Engine

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-1C3C3C?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash%20Lite-4285F4?style=for-the-badge&logo=google&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/Neon-PostgreSQL-00E599?style=for-the-badge&logo=postgresql&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Command%20Center-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**Causelith** is an end-to-end **Autonomous Business Intelligence & Multi-Agent RAG system** that bridges structured SQL warehouse telemetry with unstructured corporate knowledge.

Traditional BI dashboards stop at *"what dropped."* Causelith automatically detects statistically significant KPI anomalies in **Neon PostgreSQL**, slices affected cohorts via read-only SQL, investigates corporate policy, billing, and incident postmortem PDFs using **Hybrid Retrieval (Dense + BM25 + RRF)**, verifies evidence grounding inside a **self-correcting LangGraph loop** to prevent hallucinations, and persists audited executive action briefs directly back to the warehouse.

---

## 🎯 Verified Production Benchmark (Simulated B2B SaaS Enterprise)

Across **21,000+ warehouse records** (`$104.55M` net revenue, `$58.78M` closed-won pipeline) and **9 corporate PDFs** (including distractor compliance and roadmap documents), Causelith autonomously surfaced and verified **$3.83M+ in active revenue exposure**:

| Detected Anomaly | 30-Day Shift vs. Baseline | Exposure (USD) | Root Cause Identified | Grounded Citations | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **APAC Enterprise Conversion Collapse** | `22.83% → 6.98%` (`-69.44%`) | `$3,039,646.34` | Regional discount authority cut from 20% to 10%, routing deals to US CFO Desk (18–24 day delay) in conflict with SLA Clause 7.3 | `POL-APAC-402`, `ESC-APAC-114` | `VERIFIED (1.00)` |
| **Europe Pro Tier Discount Leakage** | `7.49% → 27.40%` (`+265.67%`) | `$308,664.44` | CRM-to-ERP connector upgrade disabled validation rule `VR-19`, enabling double stacking of `EU-AI-LAUNCH` (15%) + annual prepay (12%) | `INC-884`, `MKT-EU-209` | `VERIFIED (1.00)` |
| **North America Mid-Market Churn Surge** | `3.20% → 20.00%` (`+524.85%`) | `$486,313.20` | API Gateway v4.2 enforced a 100 req/min rate limit on `us-east-1`, breaking HubSpot/Salesforce sync for 6 days and triggering SLA Clause 4.1 exits | `PM-2026-109`, `SLA-GLB-2026` | `VERIFIED (0.99)` |

---

## 🏗️ System Architecture

```text
[Neon PostgreSQL Warehouse — 21,000+ Rows]
  ├── leads_funnel (12,000 rows)
  └── revenue_ledger (9,000 rows)
          │
          ▼  Stage 1: Deterministic 30-Day vs. 150-Day SQL Anomaly Scanner
[LangGraph Stateful Multi-Agent Orchestrator]
  ├── Node 1: SQL Slicer           ──► Read-only cohort drill-down & USD exposure
  ├── Node 2: Hybrid RAG Agent     ──► Gemini Embeddings (Cosine) + BM25 + RRF over PDFs
  ├── Node 3: Audit Verifier       ──► Grades evidence grounding (anti-hallucination gate)
  │      │
  │      ├── [Unverified / Weak] ──► Node 4: Query Rewriter ──► Loops back to Hybrid RAG (max 2 retries)
  │      └── [Grounded]          ──► Node 5: Action Synthesizer (Pydantic-validated JSON brief)
          │
          ▼  Stage 2: Warehouse Write-Back (`causal_insights` table)
[Delivery Layer]
  ├── FastAPI REST Microservice (/docs Swagger UI)
  ├── Dark-Mode Streamlit + Plotly Command Center (4 Interactive Tabs)
  └── Power BI / Excel CSV Export & Direct PostgreSQL Sync
```

---

## 🧠 Self-Correcting Anti-Hallucination Trace (Live Example)

When initial retrieval pulls off-region incident reports, the **Verifier Node** blocks hallucinated synthesis and automatically triggers the **Query Rewriter** before generating the executive report:

```text
[SQL Slicer]       Quantified exposure ($3,039,646.34) and sliced 4 cohort rows.
[Hybrid RAG]       Attempt #1 -> retrieved ['PM-2026-109', 'ESC-APAC-114', 'INC-884']
[Verifier]         Status=NEEDS RETRY (confidence=0.00) -> Docs reference NA/EU incidents, not APAC pricing delays.
[Query Rewriter]   Refined query -> 'APAC Enterprise pricing approval delay SLA discount conflict policy postmortem'
[Hybrid RAG]       Attempt #2 -> retrieved ['ESC-APAC-114', 'POL-APAC-402']
[Verifier]         Status=GROUNDED (confidence=1.00) -> Directly ties conversion drop to Finance Policy POL-APAC-402.
[Synthesizer]      Executive brief generated & persisted to Neon PostgreSQL (causal_insights).
```

---

## ✨ Core Engineering Highlights

* **Hybrid Retrieval (Dense + Sparse + RRF):** Combines **Gemini Embeddings (`gemini-embedding-001`)** L2-normalized NumPy cosine similarity with **`rank-bm25`** lexical search fused via **Reciprocal Rank Fusion (RRF)**. Preserves exact policy codes, SLA clauses, and engineering rules (`POL-APAC-402`, `VR-19`, `INC-884`) that vector-only search often misses.
* **Read-Only SQL Security Guardrails:** Every warehouse query passes through `run_read_only_query()`, enforcing `SELECT`/`WITH`-only prefixes and blocking destructive DDL/DML keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, `COPY`, `EXECUTE`).
* **Structured Output Validation:** Uses **Pydantic** models (`VerificationResult`, `ExecutiveBrief`) with fault-tolerant JSON extraction to guarantee clean database persistence and API responses.
* **High-Speed Bulk ETL:** Loads 21,000+ rows into cloud PostgreSQL in seconds using native `COPY FROM STDIN` streaming via `psycopg2`.
* **Dual Production Delivery:** Exposes full programmatic access via **FastAPI** alongside a custom dark-mode **Streamlit** command center.

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Database & ETL** | Neon PostgreSQL (Connection Pooling), SQLAlchemy, `psycopg2-binary`, Pandas, NumPy, `fpdf2` |
| **Agent Orchestration & LLM** | LangGraph, LangChain, Google Gemini API (`gemini-3.5-flash-lite`, `gemini-embedding-001`), Pydantic |
| **Hybrid Search / RAG** | Normalized NumPy Cosine Similarity, `rank-bm25`, Reciprocal Rank Fusion (RRF), `pypdf` |
| **API & Frontend** | FastAPI, Uvicorn, Streamlit (Custom Dark Theme), Plotly |

---

## 📁 Repository Structure

```text
Causelith/
├── .streamlit/
│   └── config.toml            # Dark-mode executive theme configuration
├── data/
│   ├── pdfs/                  # 9 corporate policy, SLA, billing, postmortem & distractor PDFs
│   ├── chunks.json            # Chunked document metadata & Document IDs
│   └── vector_index.npz       # Precomputed L2-normalized Gemini embedding matrix
├── src/
│   ├── __init__.py
│   ├── database.py            # Neon PostgreSQL connection, DDL, COPY bulk-loader & SQL guardrail
│   ├── generate_data.py       # Synthetic B2B SaaS generator (21k rows + 3 leaks + 9 PDFs)
│   ├── rag_engine.py          # Hybrid Dense + BM25 + RRF retrieval engine
│   ├── agent_graph.py         # SQL anomaly detector + stateful LangGraph multi-agent workflow
│   └── api.py                 # FastAPI REST microservice with Swagger docs
├── app.py                     # 4-tab dark-mode Streamlit Executive Command Center
├── requirements.txt           # Pinned project dependencies
├── .gitignore                 # Excludes .env, venv/, and cache artifacts
└── README.md
```

---

## 🖥️ Executive Command Center (`app.py`)

1. **📊 Executive BI Telemetry:** Top-level KPI strip (`Net Revenue`, `Closed-Won Pipeline`, `Active Anomalies`, `Revenue at Risk`, `Verified AI Briefs`) plus 4 interactive Plotly dark-mode visualizations across conversion, discount leakage, churn, and USD exposure.
2. **🤖 Live Multi-Agent Investigator:** Trigger the full LangGraph workflow on any detected anomaly on demand and inspect the real-time reasoning trace, SQL cohort drill-down, and verified brief.
3. **📑 Verified Causal Insights Ledger:** Reads persisted records directly from Neon PostgreSQL (`causal_insights`) with one-click CSV export ready for Power BI or Excel.
4. **💬 Executive AI Copilot:** Ad-hoc Q&A combining live SQL KPI snapshots with Hybrid RAG citations over the full corporate document library.

---

## 🔌 FastAPI REST Endpoints (`src/api.py`)

Interactive Swagger documentation is available at `/docs`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service metadata and endpoint discovery |
| `GET` | `/health` | Service health check |
| `GET` | `/api/kpis` | Live aggregate funnel and revenue KPIs from Neon PostgreSQL |
| `GET` | `/api/anomalies` | Runs real-time 30-day vs. 150-day anomaly scan across warehouse tables |
| `GET` | `/api/insights` | Returns persisted verified root-cause briefs from `causal_insights` |
| `POST` | `/api/investigate` | Runs live LangGraph multi-agent investigation for a specified `anomaly_id` |
| `POST` | `/api/ask` | Queries the Executive AI Copilot with Hybrid RAG + live SQL context |

---

## 🚀 Local Quickstart

### 1. Clone & Create Virtual Environment
```bash
git clone [https://github.com/](https://github.com/)<your-github-username>/Causelith.git
cd Causelith

python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://<user>:<password>@<neon-host>/<dbname>?sslmode=require
GOOGLE_API_KEY=<your-google-ai-studio-api-key>
```

### 3. Run the End-to-End Pipeline
```bash
# Step 1: Initialize Neon PostgreSQL tables, load 21,000 rows & generate 9 corporate PDFs
python -m src.generate_data

# Step 2: Build the Hybrid Dense + BM25 index
python -m src.rag_engine

# Step 3: Detect KPI anomalies, run LangGraph investigations & persist verified briefs
python -m src.agent_graph
```

### 4. Launch Backend & Dashboard
Open two terminal tabs inside your activated `venv`:

```bash
# Terminal 1 — FastAPI Backend ([http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs))
uvicorn src.api:app --reload --port 8000

# Terminal 2 — Streamlit Command Center (http://localhost:8501)
streamlit run app.py
```

---

## 🌐 Cloud Deployment

### Streamlit Community Cloud (Dashboard)
1. Push this repository to GitHub (ensure `.env` and `venv/` are excluded via `.gitignore`).
2. Connect the repository at [share.streamlit.io](https://share.streamlit.io) with main file path **`app.py`**.
3. Add `DATABASE_URL` and `GOOGLE_API_KEY` under **Advanced Settings → Secrets** in TOML format.

### Render (FastAPI Microservice)
1. Create a new **Web Service** on [dashboard.render.com](https://dashboard.render.com) linked to this repository.
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
4. Add `DATABASE_URL` and `GOOGLE_API_KEY` under **Environment Variables**.

### Live Links
* **Executive Dashboard (Streamlit Cloud):** `https://<your-app-name>.streamlit.app` *(update after deployment)*
* **REST API & Swagger Docs (Render):** `https://<your-service-name>.onrender.com/docs` *(update after deployment)*

---

## 👤 Author

**Rajnish Kumar** — AI Engineer & Data Analytics Specialist  
Built independently to demonstrate production-grade Agentic RAG, Hybrid Search, SQL Analytics, and Full-Stack AI deployment.