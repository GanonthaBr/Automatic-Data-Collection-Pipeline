"""
Mini-Grid Data Collection Pipeline
===================================
Agentic pipeline for mini-grid deployment data collection via web search
and PDF ingestion, with Pydantic-aligned validation and derived field computation.

Usage:
    python pipeline.py --country "Rwanda" --api_key "YOUR_GEMINI_API_KEY"
    python pipeline.py --country "Nigeria" --api_key "YOUR_GEMINI_API_KEY" --searches 5
    python pipeline.py --country "Kenya" --api_key "YOUR_GEMINI_API_KEY" --pdfs "reports/*.pdf"
    python pipeline.py --country "Kenya" --api_key "YOUR_GEMINI_API_KEY" --searches 3 --pdfs "data/kenya_report.pdf"
"""

import argparse
import csv
import glob as _glob
import json
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from ddgs import DDGS

# ── Config ────────────────────────────────────────────────────────────────────
PROVENANCE_LOG = "provenance_log.csv"
OUTPUT_FILE    = "minigrid_dataset.csv"
GEMINI_MODEL   = "gemini-2.5-pro"

DATASET_FIELDS = [
    # A. Project Identification
    "project_id", "country", "region", "project_name", "developer",
    "business_model", "status", "year_announced", "operation_year",
    "city", "state_province", "latitude", "longitude", "grid_interconnection",
    # B. Deployment Structure
    "number_of_mini_grids", "installed_or_planned",
    # C. Customer & Access Data
    "total_connections", "household_connections", "business_connections",
    "public_institution_connections", "productive_use_connections",
    "total_people_reported", "first_time_access", "improved_connection",
    "direct_or_indirect_classification", "tier_level",
    # D. Generation Capacity (kW)
    "total_installed_capacity", "total_planned_capacity", "solar_pv_capacity",
    "wind_capacity", "hydro_capacity", "biomass_capacity",
    "diesel_hfo_capacity", "other_capacity",
    # E. Storage
    "storage_included", "battery_capacity", "battery_chemistry_type",
    # F. Financing (USD)
    "total_project_cost", "equity", "debt", "grant",
    "public_financing", "private_financing", "results_based_financing_rbf",
    "blended_finance_component", "primary_investor_financier_name",
    "investment_date", "time_from_approval_to_disbursement",
    # G. Performance
    "annual_output", "delivered_energy", "load_factor",
    # H. Cross-Cutting Flags
    "climate_resilient", "productive_use_enabled",
    "gender_targeted_component", "fragile_fcv_context",
    # I. Derived Data (auto-computed)
    "capacity_per_mini_grid", "connections_per_mini_grid",
    "capacity_per_connection", "cost_per_connection", "cost_per_kw",
    "people_per_connection", "implied_investment_per_person",
    # J. Pace of Development
    "time_proposal_submission_to_site_award",
    "time_site_award_to_contract_signature",
    "time_contract_signature_to_financial_close",
    "time_purchase_order_to_equipment_arrival",
    "time_equipment_arrival_to_commissioning",
    "time_to_reach_50_plus_customer_connections",
    "regulatory_approval_timeline",
    "annual_minigrids_commissioned_country",
    # K. Enabling Environment
    "rise_score_mini_grid_sub_index", "doing_business_score_ranking",
    "policy_regulatory_framework_status", "mobile_network_coverage",
    "internet_availability", "rural_digital_payment_infrastructure",
    # L. Capacity Building
    "om_training_provided", "training_provider",
    "number_of_personnel_trained", "post_training_employment_rate",
    # Sources & Notes
    "sources", "analyst_notes", "source_language",
    # Operational
    "source_url", "data_quality_rating", "extraction_timestamp",
]

# ── Field type registries (for validation / coercion) ─────────────────────────
_FLOAT_FIELDS = {
    "latitude", "longitude",
    "total_installed_capacity", "total_planned_capacity", "solar_pv_capacity",
    "wind_capacity", "hydro_capacity", "biomass_capacity",
    "diesel_hfo_capacity", "other_capacity",
    "battery_capacity", "total_project_cost", "equity", "debt", "grant",
    "public_financing", "private_financing", "results_based_financing_rbf",
    "blended_finance_component", "time_from_approval_to_disbursement",
    "annual_output", "delivered_energy", "load_factor",
    "mobile_network_coverage", "internet_availability",
    "post_training_employment_rate",
    "rise_score_mini_grid_sub_index", "doing_business_score_ranking",
    # Derived
    "capacity_per_mini_grid", "connections_per_mini_grid",
    "capacity_per_connection", "cost_per_connection", "cost_per_kw",
    "people_per_connection", "implied_investment_per_person",
    # Pace
    "time_proposal_submission_to_site_award", "time_site_award_to_contract_signature",
    "time_contract_signature_to_financial_close", "time_purchase_order_to_equipment_arrival",
    "time_equipment_arrival_to_commissioning", "time_to_reach_50_plus_customer_connections",
    "regulatory_approval_timeline",
}

_INT_FIELDS = {
    "year_announced", "operation_year", "number_of_mini_grids",
    "total_connections", "household_connections", "business_connections",
    "public_institution_connections", "productive_use_connections",
    "total_people_reported", "investment_date", "number_of_personnel_trained",
    "annual_minigrids_commissioned_country",
}

_YN_FIELDS = {
    "grid_interconnection", "first_time_access", "improved_connection",
    "storage_included", "climate_resilient", "productive_use_enabled",
    "gender_targeted_component", "fragile_fcv_context", "om_training_provided",
}


# ── Validation & enrichment ───────────────────────────────────────────────────

def _safe_div(a, b) -> float | None:
    try:
        fa, fb = float(a), float(b)
        if fb == 0:
            return None
        return round(fa / fb, 4)
    except (TypeError, ValueError):
        return None


def validate_and_enrich(record: dict, source_url: str = "") -> dict:
    """
    Coerce LLM output to correct types, enforce Y/N vocab, compute
    Section-I derived fields, and auto-populate the sources list.
    """
    # Type coercion
    for field in _FLOAT_FIELDS:
        val = record.get(field)
        if val is not None:
            try:
                record[field] = float(val)
            except (TypeError, ValueError):
                record[field] = None

    for field in _INT_FIELDS:
        val = record.get(field)
        if val is not None:
            try:
                record[field] = int(float(val))
            except (TypeError, ValueError):
                record[field] = None

    for field in _YN_FIELDS:
        val = record.get(field)
        if val is not None:
            normalized = str(val).strip().upper()
            record[field] = normalized if normalized in ("Y", "N") else None

    # Section I — derived fields
    cap        = record.get("total_installed_capacity")
    n_grids    = record.get("number_of_mini_grids")
    conns      = record.get("total_connections")
    cost       = record.get("total_project_cost")
    people     = record.get("total_people_reported")

    record["capacity_per_mini_grid"]        = _safe_div(cap, n_grids)
    record["connections_per_mini_grid"]     = _safe_div(conns, n_grids)
    record["capacity_per_connection"]       = _safe_div(
        float(cap) * 1000 if cap is not None else None, conns
    )
    record["cost_per_connection"]           = _safe_div(cost, conns)
    record["cost_per_kw"]                   = _safe_div(cost, cap)
    record["people_per_connection"]         = _safe_div(people, conns)
    record["implied_investment_per_person"] = _safe_div(cost, people)

    # Auto-populate sources from source_url if not already set
    if source_url and not record.get("sources"):
        record["sources"] = [{"source_type": "url", "url": source_url}]

    return record


# ── PDF ingestion ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "Error: pypdf not installed. Run: pip install pypdf"
    try:
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages[:30]:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)[:8000]
    except Exception as e:
        return f"Error reading PDF: {e}"


# ── Web helpers ───────────────────────────────────────────────────────────────

def fetch_webpage(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
        resp = requests.get(url.strip(), headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:4000]
    except Exception as e:
        return f"Error fetching page: {e}"


def log_provenance(country, query, source_url, quality, notes=""):
    row = {
        "timestamp": datetime.now().isoformat(), "country": country,
        "search_query": query, "source_url": source_url,
        "data_quality_rating": quality, "notes": notes,
    }
    file_exists = os.path.isfile(PROVENANCE_LOG)
    with open(PROVENANCE_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Quality rating ────────────────────────────────────────────────────────────

def compute_quality_rating(record: dict) -> int:
    """
    Scoring rules:
      Core identity  (project_name, country, developer)               — 1 pt each
      Capacity       (total_installed_capacity or solar_pv_capacity)  — 1 pt
      Connections    (total_connections)                               — 1 pt
      Contextual     (operation_year, status)                         — 1 pt each

    Total possible: 7 points
      6-7 → 3 (High)   4-5 → 2 (Medium)   0-3 → 1 (Low)
    """
    def has_value(v):
        return v is not None and str(v).strip() not in ("", "Unknown", "null", "None")

    capacity_present = (
        has_value(record.get("total_installed_capacity"))
        or has_value(record.get("solar_pv_capacity"))
    )
    score = sum([
        has_value(record.get("project_name")),
        has_value(record.get("country")),
        has_value(record.get("developer")),
        capacity_present,
        has_value(record.get("total_connections")),
        has_value(record.get("operation_year")),
        has_value(record.get("status")),
    ])
    if score >= 6:
        return 3
    elif score >= 4:
        return 2
    else:
        return 1


# ── Save ──────────────────────────────────────────────────────────────────────

def save_to_dataset(record: dict) -> str:
    record["extraction_timestamp"] = datetime.now().isoformat()
    record["data_quality_rating"] = compute_quality_rating(record)
    for field in ("sources", "analyst_notes"):
        val = record.get(field)
        if isinstance(val, (list, dict)):
            record[field] = json.dumps(val, ensure_ascii=False)
    for field in DATASET_FIELDS:
        record.setdefault(field, "")
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: record[k] for k in DATASET_FIELDS})
    return f"Saved: {record.get('project_name', 'Unknown')} ({record.get('country', '')})"


# ── LLM extraction ────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """\
You are a structured data extraction assistant for an energy access research project.

Given the text below about mini-grid projects in {country}, extract ALL mini-grid \
deployments mentioned. For each project, return a JSON array of objects with these fields \
(use null for any field not found in the text):

A. Project Identification:
- project_name (string)
- country (string)
- region (string)
- city (string)
- state_province (string)
- developer (string — company/org developing the project)
- business_model (string)
- status (string — e.g. "Operational", "Planned", "Under Construction")
- year_announced (integer)
- operation_year (integer — year commissioned/operational)
- latitude (float)
- longitude (float)
- grid_interconnection ("Y" or "N")

B. Deployment Structure:
- number_of_mini_grids (integer)
- installed_or_planned (string)

C. Customer & Access Data:
- total_connections (integer)
- household_connections (integer)
- business_connections (integer)
- public_institution_connections (integer)
- productive_use_connections (integer)
- total_people_reported (integer)
- first_time_access ("Y" or "N")
- improved_connection ("Y" or "N")
- direct_or_indirect_classification (string)
- tier_level (string — MTF tier)

D. Generation Capacity (kW — convert MW×1000, kVA×0.8):
- total_installed_capacity (float)
- total_planned_capacity (float)
- solar_pv_capacity (float)
- wind_capacity (float)
- hydro_capacity (float)
- biomass_capacity (float)
- diesel_hfo_capacity (float)
- other_capacity (float)

E. Storage:
- storage_included ("Y" or "N")
- battery_capacity (float — kWh)
- battery_chemistry_type (string — e.g. "lithium-ion (LFP, NMC)", "lead-acid", "other")

F. Financing (USD):
- total_project_cost (float)
- equity (float)
- debt (float)
- grant (float)
- public_financing (float)
- private_financing (float)
- results_based_financing_rbf (float)
- blended_finance_component (float)
- primary_investor_financier_name (string)
- investment_date (integer — year)
- time_from_approval_to_disbursement (float — months)

G. Performance:
- annual_output (float — kWh/year)
- delivered_energy (float — kWh)
- load_factor (float — %)

H. Cross-Cutting Flags:
- climate_resilient ("Y" or "N")
- productive_use_enabled ("Y" or "N")
- gender_targeted_component ("Y" or "N")
- fragile_fcv_context ("Y" or "N")

K. Enabling Environment (if country-level data present):
- policy_regulatory_framework_status (string — e.g. "dedicated minigrid regulation")
- mobile_network_coverage (float — %)
- internet_availability (float — %)
- rural_digital_payment_infrastructure (string — "widely available", "limited", or "not available")

L. Capacity Building:
- om_training_provided ("Y" or "N")
- training_provider (string)
- number_of_personnel_trained (integer)
- post_training_employment_rate (float — %)

Additional instructions:
- source_language: detect the language of the source text and set to "en", "fr", "pt", or "other"
- Regardless of source language, ALL extracted field values must be in English
  (e.g. status: "Operational" not "Opérationnel", "Planned" not "Prévu")
- For numeric fields in French/Portuguese documents, handle comma decimal separators
  and space thousand separators correctly (e.g. "1 500,5 kW" → 1500.5)

Return ONLY a valid JSON array. No explanation, no markdown, no extra text.

TEXT:
{text}
"""


def extract_structured_data(llm, country: str, text: str, source_url: str,
                             max_retries: int = 3, text_limit: int = 3500) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(country=country, text=text[:text_limit])
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            raw = response.content.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            records = json.loads(raw)
            if not isinstance(records, list):
                records = [records]
            for r in records:
                r["country"] = country
                r["source_url"] = source_url
                validate_and_enrich(r, source_url)
            return records
        except json.JSONDecodeError:
            print(f"  [!] JSON parse error on attempt {attempt+1}, retrying...")
            time.sleep(2)
        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower() or "429" in error_str:
                print(f"  [~] Rate limited. Waiting 30s (attempt {attempt+1}/{max_retries})...")
                time.sleep(30)
            else:
                print(f"  [!] Extraction error: {e}")
                return []
    print(f"  [!] Failed after {max_retries} retries.")
    return []


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(country: str, api_key: str, num_searches: int = 3,
                 pdf_paths: list[str] | None = None):
    print(f"\n{'='*60}")
    print(f"  Mini-Grid Data Collection Pipeline  |  Powered by Gemini")
    print(f"  Country : {country}")
    print(f"  Model   : {GEMINI_MODEL}")
    print(f"{'-'*70}\n")

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1, google_api_key=api_key)

    FRANCOPHONE = {
        "chad", "niger", "mali", "senegal", "burkina faso", "guinea",
        "cameroon", "madagascar", "benin", "togo", "côte d'ivoire",
        "cote d'ivoire", "drc", "congo", "mauritania", "comoros",
    }
    is_francophone = country.lower() in FRANCOPHONE

    queries = [
        f"{country} mini-grid solar deployment projects MW connections",
        f"{country} off-grid electrification rural energy access report",
        f"{country} mini-grid regulatory framework utility energy ministry",
        f"site:irena.org OR site:esmap.org {country} mini-grid",
        f"{country} mini-grid private developer BBOXX Engie PowerGen Ignite",
    ]
    if is_francophone:
        print(f"  [i] Francophone country detected — adding French queries")
        queries += [
            f"{country} mini-réseau solaire déploiement électrification rurale",
            f"{country} accès énergie hors-réseau rapport ministère",
            f"{country} mini-grid électricité villages projet développement",
        ]

    all_records = []
    searched_urls = set()

    # ── Web search ────────────────────────────────────────────────────────────
    for i, query in enumerate(queries[:num_searches], 1):
        print(f"[{i}/{num_searches}] Searching: {query}")
        try:
            ddgs_results = list(DDGS().text(query, max_results=5))
            print(f"  → Got {len(ddgs_results)} search results")

            urls = [r["href"] for r in ddgs_results
                    if r.get("href") and r["href"] not in searched_urls][:3]
            combined_snippets = " ".join(
                f"{r.get('title','')} {r.get('body','')}" for r in ddgs_results
            )

            if urls:
                snippet_source = urls[0]
                records = extract_structured_data(llm, country, combined_snippets, snippet_source)
                if records:
                    print(f"  → Extracted {len(records)} record(s) from snippets")
                    all_records.extend(records)
                    log_provenance(country, query, snippet_source, 2, "From search snippets")

            for url in urls:
                searched_urls.add(url)
                print(f"  → Fetching: {url[:70]}...")
                page_text = fetch_webpage(url)
                if "Error" not in page_text and len(page_text) > 200:
                    page_records = extract_structured_data(llm, country, page_text, url)
                    if page_records:
                        print(f"     → Extracted {len(page_records)} record(s)")
                        all_records.extend(page_records)
                        quality = max(
                            (r.get("data_quality_rating", 1) for r in page_records), default=1
                        )
                        log_provenance(country, query, url, quality, "From full page fetch")
                time.sleep(1)

        except Exception as e:
            print(f"  [!] Search error: {e}")
        time.sleep(1)

    # ── PDF ingestion ─────────────────────────────────────────────────────────
    if pdf_paths:
        print(f"\n[PDF] Processing {len(pdf_paths)} file(s)...")
        for pdf_path in pdf_paths:
            print(f"  → Reading: {os.path.basename(pdf_path)}")
            text = extract_text_from_pdf(pdf_path)
            if text.startswith("Error"):
                print(f"     [!] {text}")
                continue
            source_ref = f"file://{os.path.abspath(pdf_path)}"
            # Use larger text window for PDFs since they are denser documents
            records = extract_structured_data(
                llm, country, text, source_ref, text_limit=6000
            )
            if records:
                print(f"     → Extracted {len(records)} record(s)")
                for r in records:
                    r["sources"] = [
                        {"source_type": "pdf", "file_path": os.path.abspath(pdf_path)}
                    ]
                all_records.extend(records)
                log_provenance(country, f"PDF:{os.path.basename(pdf_path)}",
                               source_ref, 3, "From PDF ingestion")

    # ── Deduplicate ───────────────────────────────────────────────────────────
    seen = set()
    unique_records = []
    for r in all_records:
        key = (
            (r.get("project_name") or "").lower().strip(),
            (r.get("country") or "").lower().strip(),
            (r.get("city") or "").lower().strip(),
        )
        if key not in seen and (r.get("project_name") or "Unknown") != "Unknown":
            seen.add(key)
            unique_records.append(r)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Saving {len(unique_records)} unique record(s) to {OUTPUT_FILE}...")
    for record in unique_records:
        print(f"  {save_to_dataset(record)}")

    if unique_records:
        df = pd.DataFrame(unique_records)
        cols = [
            "project_name", "country", "city", "developer", "status",
            "total_installed_capacity", "solar_pv_capacity",
            "total_connections", "operation_year",
            "total_project_cost", "cost_per_kw", "data_quality_rating",
        ]
        display_cols = [c for c in cols if c in df.columns]
        print(f"\n{'─'*60}")
        print(f"EXTRACTED DATASET — {country.upper()}")
        print(df[display_cols].to_string(index=False))
    else:
        print("\n[!] No structured records extracted.")

    print(f"\n{'-'*60}")
    print(f"  Done! Output files:")
    print(f"  - {OUTPUT_FILE}       (extracted dataset)")
    print(f"  - {PROVENANCE_LOG}    (search audit log)")
    print(f"{'-'*60}\n")
    return unique_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-Grid Agentic Data Collection Pipeline")
    parser.add_argument("--country",  type=str, required=True, help="Country to research")
    parser.add_argument("--api_key",  type=str, required=True, help="Gemini API key")
    parser.add_argument("--searches", type=int, default=3,
                        help="Number of web search queries (default: 3)")
    parser.add_argument("--pdfs",     type=str, default=None,
                        help="Glob pattern or path for PDF files to ingest "
                             "(e.g. 'reports/*.pdf' or 'data/kenya.pdf')")
    args = parser.parse_args()

    resolved_pdfs = None
    if args.pdfs:
        resolved_pdfs = _glob.glob(args.pdfs)
        if not resolved_pdfs:
            print(f"[!] No PDF files matched: {args.pdfs}")

    run_pipeline(args.country, args.api_key, args.searches, resolved_pdfs)
