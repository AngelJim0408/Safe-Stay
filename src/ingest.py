"""Download incidents from Socrata with stable paging, retries and resume.

Each page is written to data/raw/page_00000.json; existing pages are skipped,
so a failure at page 40 does not restart from zero.
Finally all pages are combined into one raw Parquet file (all columns as strings).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

import config


def _where_clause() -> str:
    return (f"{config.COL_DATETIME} >= '{config.START_DATE}T00:00:00' "
            f"AND {config.COL_DATETIME} <= '{config.END_DATE}T23:59:59'")


def _headers() -> dict:
    h = {"Accept": "application/json"}
    if config.SOCRATA_APP_TOKEN:
        h["X-App-Token"] = config.SOCRATA_APP_TOKEN
    if config.CONTACT_EMAIL:
        h["User-Agent"] = f"safestay-student-project ({config.CONTACT_EMAIL})"
    return h


def fetch_page(offset: int) -> list:
    params = {
        "$select": ",".join(config.SELECT_COLUMNS),
        "$where": _where_clause(),
        "$order": config.ORDER_COLUMN,       # stable order => no overlap/skips
        "$limit": config.PAGE_SIZE,
        "$offset": offset,
    }
    for attempt in range(config.MAX_RETRIES):
        try:
            r = requests.get(config.SOCRATA_URL, params=params, headers=_headers(),
                             timeout=config.REQUEST_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            wait = min(2 ** attempt, 60)
            print(f"  offset {offset}: {exc}; retry {attempt + 1}/{config.MAX_RETRIES} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Giving up at offset {offset}; rerun to resume.")


def download_pages() -> int:
    config.ensure_dirs()
    page = 0
    while True:
        path = config.RAW_DIR / f"page_{page:05d}.json"
        if path.exists():
            rows = json.loads(path.read_text())
        else:
            rows = fetch_page(page * config.PAGE_SIZE)
            path.write_text(json.dumps(rows))
            print(f"page {page}: {len(rows):,} rows")
        if len(rows) < config.PAGE_SIZE:       # last page
            return page + 1
        page += 1


def build_raw_parquet() -> pd.DataFrame:
    n_pages = download_pages()
    frames = [pd.DataFrame(json.loads((config.RAW_DIR / f"page_{i:05d}.json").read_text()))
              for i in range(n_pages)]
    df = pd.concat(frames, ignore_index=True).astype("string")
    df.to_parquet(config.RAW_PARQUET, index=False)
    print(f"raw rows: {len(df):,} -> {config.RAW_PARQUET}")
    return df


def smoke_test(n: int = 5) -> None:
    """One small request: confirm the dataset ID and print the real column names."""
    r = requests.get(config.SOCRATA_URL, params={"$limit": n}, headers=_headers(),
                     timeout=config.REQUEST_TIMEOUT)
    r.raise_for_status()
    rows = r.json()
    print("status", r.status_code, "| columns:", sorted(rows[0].keys()) if rows else "none")


if __name__ == "__main__":
    smoke_test()
