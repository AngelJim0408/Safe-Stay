"""Task 2: python run_task2.py --month 7 [--no-llm]"""
import argparse
import calendar
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import config  # noqa: E402
import grounding  # noqa: E402
import llm  # noqa: E402
import outlook  # noqa: E402

SYSTEM = ("You write short, traveller-friendly neighbourhood safety summaries. "
          "Use ONLY the numbers in the data provided. Do not invent or compute new numbers.")


def main(month: int, use_llm: bool = True) -> None:
    config.ensure_dirs()
    name = calendar.month_name[month]
    level = outlook.compute_outlook(outlook.load_table(), month)
    rows = outlook.select_rows(level).round(3)
    rows_csv = rows.to_csv(index=False)
    rows.to_csv(config.OUTPUT_DIR / f"t2_outlook_{month:02d}.csv", index=False)

    out_md = config.OUTPUT_DIR / f"t2_summary_{month:02d}.md"
    summary = None
    if use_llm:
        prompt = (f"Month: {name}. Columns: avg_incidents (per month, 2022-2024), persons, "
                  f"property, share_of_city, seasonal_ratio, list_name.\n\n{rows_csv}\n\n"
                  "Write a short summary for a traveller.")
        for attempt in range(config.MAX_LLM_RETRIES):
            try:
                text = llm.generate(SYSTEM, prompt)
            except RuntimeError as exc:
                print(exc)
                break
            ok, bad = grounding.verify(text, json.loads(rows.to_json(orient="records")))
            if ok:
                summary = text
                break
            print(f"attempt {attempt + 1}: unmatched numbers {bad}")
    body = f"# {name} outlook\n\n"
    body += (summary + "\n\n" if summary else
             "_Summary unavailable (failed number check or LLM error); table only._\n\n")
    body += "## Data\n\n" + rows.to_markdown(index=False) + "\n"
    out_md.write_text(body, encoding="utf-8")
    print("wrote", out_md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    main(a.month, use_llm=not a.no_llm)
