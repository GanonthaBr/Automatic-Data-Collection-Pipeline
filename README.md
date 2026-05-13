# Mini-Grid Data Collection Pipeline
### Agentic AI for Energy Access Research

This pipeline simulates the desk research and structured data extraction workflow
described in the Graduate RA position for mini-grid data collection.



## What it does



| JD Responsibility | Pipeline Implementation |
|---|---|
| Desk Research & Data Collection | DuckDuckGo search across targeted queries |
| LLM-based extraction | llama-3.3-70b-versatile parses documents into structured JSON |
| Multilingual Data Extraction | llama handles French/English documents natively |
| Source Documentation | Auto-logs every search + URL to `provenance_log.csv` |
| Structured database population | Appends clean records to `minigrid_dataset.csv` |



## Setup

### 1. Install dependencies
```bash
pip install langchain langchain-google-genai google-generativeai \
            langchain-community duckduckgo-search \
            beautifulsoup4 requests pandas openpyxl
```





## Usage

```bash
# Basic usage
python pipeline.py --country "Rwanda" --api_key "API_KEY"

# Run more searches for richer results
python pipeline.py --country "Niger" --api_key "API_KEY" --searches 5

# Try multiple countries
python pipeline.py --country "Tanzania" --api_key "API_KEY"
python pipeline.py --country "Senegal" --api_key "API_KEY"
```



## Output files

### `minigrid_dataset.csv`
Structured dataset with one row per mini-grid project:

| Field | Description |
|---|---|
| country | Country researched |
| project_name | Name of mini-grid project |
| developer_operator | Company/org operating it |
| capacity_kw | Installed capacity in kW |
| num_connections | Number of households/connections |
| location | Village/region |
| year_commissioned | Year project went live |
| technology_type | solar / hybrid / hydro |
| source_url | Where data was found |
| data_quality_rating | 1=Low, 2=Medium, 3=High |
| notes | Caveats or extra context |
| extraction_timestamp | When it was extracted |

### `provenance_log.csv`
Audit trail of every search query and URL accessed — mirrors the
"source provenance and data quality ratings" requirement in the JD.



## Data quality ratings

| Rating | Meaning |
|---|---|
| 1 - Low | Vague mention, missing most fields |
| 2 - Medium | Partial data, some fields confirmed |
| 3 - High | Complete record, multiple fields verified |



## Pipeline flow

```
Input: Country name
       ↓
[1] Generate targeted search queries
       ↓
[2] DuckDuckGo search → extract snippet text
       ↓
[3] Fetch top URLs → extract full page text
       ↓
[4] LLM → parse text → structured JSON records
       ↓
[5] Deduplicate records
       ↓
[6] Save to minigrid_dataset.csv
[7] Log to provenance_log.csv
```
