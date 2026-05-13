"""
Mini-Grid Data Collection Pipeline
===================================
Agentic pipeline that simulates desk research & structured data extraction
for mini-grid deployment data — based on the RA job responsibilities.

Usage:
    python pipeline.py --country "Rwanda" --api_key "YOUR_GROQ_API_KEY"
    python pipeline.py --country "Niger" --api_key "YOUR_GROQ_API_KEY" --searches 5
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from ddgs import DDGS

# ── Config ───────────────────────────────────────────────────────────────────
PROVENANCE_LOG = "provenance_log.csv"
OUTPUT_FILE    = "minigrid_dataset.csv"
GROQ_MODEL     = "llama-3.3-70b-versatile"

DATASET_FIELDS = [
    "country", "project_name", "developer_operator", "capacity_kw",
    "num_connections", "location", "year_commissioned", "technology_type",
    "source_url", "data_quality_rating", "notes", "extraction_timestamp",
]



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



def compute_quality_rating(record: dict) -> int:
    """
    Deterministic data quality rating based on field completeness.

    Scoring rules:
      Core fields (project_name, location, developer_operator) — 1 point each
      Quantitative fields (capacity_kw, num_connections)        — 1 point each
      Contextual fields (year_commissioned, technology_type)    — 1 point each

    Total possible: 7 points
      6-7 → 3 (High)   — most fields present and specific
      3-5 → 2 (Medium) — partial data
      0-2 → 1 (Low)    — vague or mostly missing
    """
    def has_value(v):
        return v is not None and str(v).strip() not in ("", "Unknown", "null", "None")

    score = sum([
        has_value(record.get("project_name")),
        has_value(record.get("location")),
        has_value(record.get("developer_operator")),
        has_value(record.get("capacity_kw")),
        has_value(record.get("num_connections")),
        has_value(record.get("year_commissioned")),
        has_value(record.get("technology_type")),
    ])

    if score >= 6:
        return 3   # High
    elif score >= 3:
        return 2   # Medium
    else:
        return 1   # Low


def save_to_dataset(record: dict) -> str:
    record["extraction_timestamp"] = datetime.now().isoformat()
    # Overwrite LLM-assigned rating with deterministic rule-based score
    record["data_quality_rating"] = compute_quality_rating(record)
    for field in DATASET_FIELDS:
        record.setdefault(field, "")
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: record[k] for k in DATASET_FIELDS})
    return f"Saved: {record.get('project_name', 'Unknown')} ({record.get('country', '')})"


# ── LLM Extraction 

EXTRACTION_PROMPT = """\
You are a structured data extraction assistant for an energy access research project.

Given the text below about mini-grid projects in {country}, extract ALL mini-grid \
deployments mentioned. For each project, return a JSON array of objects with these fields:
- project_name (string, or "Unknown" if not found)
- developer_operator (string, or "Unknown")
- capacity_kw (number or null)
- num_connections (number or null)
- location (string, or "Unknown")
- year_commissioned (number or null)
- technology_type (e.g. "solar", "hybrid", "hydro", or "Unknown")
- notes (any caveats or extra context)

Return ONLY a valid JSON array. No explanation, no markdown, no extra text.

TEXT:
{text}
"""


def extract_structured_data(llm, country, text, source_url, max_retries=3):
    prompt = EXTRACTION_PROMPT.format(country=country, text=text[:3500])
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


# ── Main Pipeline 

def run_pipeline(country: str, api_key: str, num_searches: int = 3):
    print(f"\n{'='*60}")
    print(f"  Mini-Grid Data Collection Pipeline  |  Powered by Groq")
    print(f"  Country : {country}")
    print(f"  Model   : {GROQ_MODEL}")
    print(f"{'-'*70}\n")

    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, groq_api_key=api_key)

    # Francophone countries
    FRANCOPHONE = {
        "chad", "niger", "mali", "senegal", "burkina faso", "guinea",
        "cameroon", "madagascar", "benin", "togo", "côte d'ivoire",
        "cote d'ivoire", "drc", "congo", "mauritania", "comoros",
    }
    is_francophone = country.lower() in FRANCOPHONE

    # English queries (used for all countries)
    queries = [
        f"{country} mini-grid solar deployment projects MW connections",
        f"{country} off-grid electrification rural energy access report",
        f"{country} mini-grid regulatory framework utility energy ministry",
        f"site:irena.org OR site:esmap.org {country} mini-grid",
        f"{country} mini-grid private developer BBOXX Engie PowerGen Ignite",
    ]

    # French queries added for Francophone countries
    if is_francophone:
        print(f"  [i] Francophone country detected — adding French queries")
        queries += [
            f"{country} mini-réseau solaire déploiement électrification rurale",
            f"{country} accès énergie hors-réseau rapport ministère",
            f"{country} mini-grid électricité villages projet développement",
        ]

    all_records = []
    searched_urls = set()

    for i, query in enumerate(queries[:num_searches], 1):
        print(f"[{i}/{num_searches}] Searching: {query}")
        try:
            # Use DDGS for structured results that include real URLs
            ddgs_results = list(DDGS().text(query, max_results=5))
            print(f"  → Got {len(ddgs_results)} search results")

            # Extract URLs and text snippets from structured results
            urls = [r["href"] for r in ddgs_results if r.get("href") and r["href"] not in searched_urls][:3]
            combined_snippets = " ".join(
                f"{r.get('title','')} {r.get('body','')}" for r in ddgs_results
            )

            # Extract from combined snippets, attributed to first real URL
            if urls:
                snippet_source = urls[0]
                records = extract_structured_data(llm, country, combined_snippets, snippet_source)
                if records:
                    print(f"  → Extracted {len(records)} record(s) from snippets")
                    all_records.extend(records)
                    log_provenance(country, query, snippet_source, 2, "From search snippets")

            # Fetch full pages and extract more records
            for url in urls:
                searched_urls.add(url)
                print(f"  → Fetching: {url[:70]}...")
                page_text = fetch_webpage(url)
                if "Error" not in page_text and len(page_text) > 200:
                    page_records = extract_structured_data(llm, country, page_text, url)
                    if page_records:
                        print(f"     → Extracted {len(page_records)} record(s)")
                        all_records.extend(page_records)
                        quality = max((r.get("data_quality_rating", 1) for r in page_records), default=1)
                        log_provenance(country, query, url, quality, "From full page fetch")
                time.sleep(1)

        except Exception as e:
            print(f"  [!] Search error: {e}")
        time.sleep(1)

    # Deduplicate
    seen = set()
    unique_records = []
    for r in all_records:
        key = (r.get("project_name", "").lower().strip(), r.get("location", "").lower().strip())
        if key not in seen and r.get("project_name", "Unknown") != "Unknown":
            seen.add(key)
            unique_records.append(r)

    # Save
    print(f"\n{'─'*60}")
    print(f"Saving {len(unique_records)} unique record(s) to {OUTPUT_FILE}...")
    for record in unique_records:
        print(f" {save_to_dataset(record)}")

    if unique_records:
        df = pd.DataFrame(unique_records)
        cols = ["project_name", "developer_operator", "capacity_kw",
                "num_connections", "location", "year_commissioned",
                "technology_type", "data_quality_rating"]
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
    parser.add_argument("--api_key",  type=str, required=True, help="Groq API key (console.groq.com)")
    parser.add_argument("--searches", type=int, default=3, help="Number of search queries (default: 3)")
    args = parser.parse_args()
    run_pipeline(args.country, args.api_key, args.searches)