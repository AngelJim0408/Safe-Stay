"""Single source of truth for every constant in the pipeline.

Edit column names / dataset ID HERE after checking Appendix A and one test request.
Secrets are loaded once here and read from os.environ; never print them.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # one JSON page per file (resumable ingest)
RAW_PARQUET = DATA_DIR / "raw" / "incidents_raw.parquet"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
SPARK_WAREHOUSE = ROOT / "spark-warehouse"

NEIGHBOURHOOD_MONTH_PARQUET = OUTPUT_DIR / "neighbourhood_month.parquet"
T1_SUMMARY_TXT = OUTPUT_DIR / "t1_summary.txt"
ROW_LOG_JSON = PROCESSED_DIR / "row_counts.json"

# ---- Socrata / DataSF ------------------------------------------------------
# VERIFY against Appendix A. "Police Department Incident Reports: 2018 to Present".
SOCRATA_DOMAIN = "data.sfgov.org"
DATASET_ID = "wg3w-h783"
SOCRATA_URL = f"https://{SOCRATA_DOMAIN}/resource/{DATASET_ID}.json"
PAGE_SIZE = 50_000
ORDER_COLUMN = "row_id"               # must be unique and stable for paging
REQUEST_TIMEOUT = 120
MAX_RETRIES = 6

# ---- date range ------------------------------------------------------------
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
YEARS = [2022, 2023, 2024]

# ---- column names (VERIFY with one small request) --------------------------
COL_INCIDENT_ID = "incident_id"
COL_ROW_ID = "row_id"
COL_DATETIME = "incident_datetime"    # fall back to "incident_date" if absent
COL_CATEGORY = "incident_category"
COL_SUBCATEGORY = "incident_subcategory"
COL_DESCRIPTION = "incident_description"
COL_NEIGHBOURHOOD = "analysis_neighborhood"
COL_LAT = "latitude"
COL_LON = "longitude"

# Columns to request from the API (keeps pages small)
SELECT_COLUMNS = [
    COL_INCIDENT_ID, COL_ROW_ID, COL_DATETIME, COL_CATEGORY,
    COL_SUBCATEGORY, COL_DESCRIPTION, COL_NEIGHBOURHOOD, COL_LAT, COL_LON,
]

# ---- "inside the city" bounding box (San Francisco, incl. Farallon excluded) -
LAT_MIN, LAT_MAX = 37.70, 37.84
LON_MIN, LON_MAX = -123.02, -122.35

# ---- Task 2 ----------------------------------------------------------------
OUTLOOK_MONTHS = [7, 1]               # at least two different months
MIN_MONTHLY_AVG_FOR_RATIO = 50.0      # cutoff so tiny neighbourhoods don't dominate the ratio list
SEASONAL_BASELINE = "all_months"      # "all_months" (36 months) or "same_12_months"
TOP_N = 5

# ---- secrets (names only; values come from .env) ---------------------------
SOCRATA_APP_TOKEN = os.environ.get("SOCRATA_APP_TOKEN", "")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")
LLM_KEYS = {
    "groq": os.environ.get("GROQ_API_KEY", ""),
    "gemini": os.environ.get("GEMINI_API_KEY", ""),
    "openrouter": os.environ.get("OPENROUTER_API_KEY", ""),
}
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
MAIN_LLM = "groq"                     # pick your main provider
LLM_FALLBACK_ORDER = ["gemini", "openrouter"]
MAX_LLM_RETRIES = 3
# Set to models your keys can actually use (check each provider's model list).
LLM_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.5-flash-lite",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
}


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
