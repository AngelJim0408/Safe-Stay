
import json
import re
import statistics
import time
from datetime import datetime, timezone

import config
import llm
import outlook

CACHE_DIR = config.DATA_DIR / "cache" / "llm"
MIN_SECONDS_BETWEEN_CALLS = 1.1          
RATE_LIMIT_WAIT = 65                     
BUSY_WAIT = 20                           
_last_call = {}                          

SYSTEM = (
    "You are helping compare large language model knowledge with official police data. "
    "Answer with JSON only, no prose and no markdown. "
    "Rate every neighbourhood you are given, and only those neighbourhoods, spelled exactly as given."
)

JSON_SPEC = (
    'Return exactly this JSON shape:\n'
    '{"neighbourhoods": [{"name": "<neighbourhood>", "rating": <integer 1-5>, '
    '"incidents_per_month": <number>}, ...]}\n'
    "rating: 1 = fewest reported incidents, 5 = most reported incidents.\n"
    "incidents_per_month: your estimate of the average number of police incident reports "
    "per month in that neighbourhood in 2022-2024."
)


#  real figures
def real_monthly_averages():
    """Average incidents per month per neighbourhood over all 36 months (2022-2024).

    Uses the Task 2 helper that fills missing neighbourhood-months with 0, so a
    neighbourhood with a quiet month is not over-estimated.
    """
    full = outlook.complete_grid(outlook.load_table())
    n_months = 12 * len(config.YEARS)
    real = (full.groupby("neighbourhood")["total_incidents"].sum() / n_months).rename("real_monthly_avg")
    real = real.reset_index().sort_values("real_monthly_avg", ascending=False, ignore_index=True)
    real["real_rank"] = range(1, len(real) + 1)
    return real


#  prompts
def build_prompt(names, data_csv=None):
    city = "San Francisco"
    lines = [f"City: {city}. Neighbourhoods (San Francisco Analysis Neighborhoods), "
             f"{len(names)} in total:", "; ".join(names), ""]
    if data_csv is None:
        lines.append("You have no data for this task. Use only what you already know.")
    else:
        lines += ["Here are the official average monthly incident counts for 2022-2024 "
                  "(San Francisco Police Department incident reports). Base your answer "
                  "on this data only.", "", data_csv]
    lines += ["", JSON_SPEC]
    return "\n".join(lines)


#  cached calls
def _cache_path(provider, mode):
    return CACHE_DIR / f"t3_{provider}_{mode}.json"


def _throttle(provider):
    last = _last_call.get(provider)
    if last is not None:
        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    _last_call[provider] = time.monotonic()


def ask(provider, mode, user_prompt, names, refresh=False, offline=False):
    """Ask one provider (no fallback), validate, retry with feedback if invalid.

    Every response is saved with the time it was received. A cached, valid answer
    is reused unless refresh=True.
    """
    path = _cache_path(provider, mode)
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        same_model = cached.get("model") == config.LLM_MODELS[provider]
        if not same_model and not offline:
            print(f"  cached answer is from {cached.get('model')}, but config uses "
                  f"{config.LLM_MODELS[provider]}: asking again", flush=True)
        elif cached.get("valid") or offline:
            cached["from_cache"] = True
            return cached
    if offline:
        raise RuntimeError(f"no cached answer for {provider}/{mode} and --offline was given")

    record = {"provider": provider, "model": config.LLM_MODELS[provider], "mode": mode,
              "system": SYSTEM, "user": user_prompt, "attempts": [], "valid": False}
    prompt = user_prompt
    for attempt in range(config.MAX_LLM_RETRIES):
        _throttle(provider)
        try:
            text = llm._call(provider, SYSTEM, prompt)   
            error = None
        except Exception as exc:  
            text, error = None, str(exc)
        received = datetime.now(timezone.utc).isoformat(timespec="seconds")
        parsed, problems = (None, [error]) if error else parse_and_validate(text, names)
        record["attempts"].append({"received_at": received, "response_text": text,
                                   "problems": problems})
        if parsed is not None and not problems:
            record.update(valid=True, parsed=parsed, retrieved_at=received)
            break
        if error:
            if "HTTP 429" in error:    # rate limit 
                print(f"  {provider}: rate limit reached, waiting {RATE_LIMIT_WAIT} s before retrying ...",
                      flush=True)
                time.sleep(RATE_LIMIT_WAIT)
                continue
            if "HTTP 4" in error:      
                break
           
            wait = BUSY_WAIT * (attempt + 1)
            print(f"  {provider}: service busy or slow ({error[:60]}...), retrying in {wait} s ...",
                  flush=True)
            time.sleep(wait)
            continue
        prompt = (user_prompt + "\n\nYour previous answer was not valid: "
                  + "; ".join(problems[:5]) + ". Reply again with the JSON only.")
    if not record["valid"]:
        last = record["attempts"][-1]
        record["retrieved_at"] = last["received_at"]
        record["parsed"] = None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    record["from_cache"] = False
    return record


#  JSON validation
def _extract_json(text):
    """Accept bare JSON or JSON inside ``` fences; return the outermost object."""
    if text is None:
        raise ValueError("empty response")
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(t[start:end + 1])


def parse_and_validate(text, names):
    """Return (list_of_entries, problems). Structural problems make the answer invalid.

    Unknown or missing neighbourhood names are NOT treated as invalid JSON: they are
    measured later as 'invented' and 'skipped', as the task asks.
    """
    try:
        obj = _extract_json(text)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, [f"not valid JSON ({exc})"]
    items = obj.get("neighbourhoods") if isinstance(obj, dict) else None
    if not isinstance(items, list) or not items:
        return None, ['top-level key "neighbourhoods" must be a non-empty list']
    problems, entries = [], []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            problems.append(f"item {i} is not an object")
            continue
        name, rating, est = it.get("name"), it.get("rating"), it.get("incidents_per_month")
        if not isinstance(name, str) or not name.strip():
            problems.append(f"item {i}: name missing")
            continue
        if isinstance(rating, bool) or not isinstance(rating, (int, float)) \
                or float(rating) != int(rating) or not 1 <= int(rating) <= 5:
            problems.append(f"{name}: rating must be an integer 1-5")
            continue
        if isinstance(est, bool) or not isinstance(est, (int, float)) or est < 0:
            problems.append(f"{name}: incidents_per_month must be a number >= 0")
            continue
        entries.append({"name": name.strip(), "rating": int(rating),
                        "incidents_per_month": float(est)})
    return entries, problems


# name matching
def norm(name):
    """'Financial District/South Beach', 'Financial District & South Beach' and
    'financial district and south beach' all normalise to the same key."""
    s = name.lower()
    s = re.sub(r"&|/|\band\b", " ", s)
    s = re.sub(r"\bst\b\.?", "street", s)
    return re.sub(r"[^a-z0-9]", "", s)


def match_names(entries, real_names):
    """Map LLM entries to real neighbourhood names. Returns (matched dict, invented list)."""
    lookup = {norm(n): n for n in real_names}
    matched, invented = {}, []
    for e in entries:
        real = lookup.get(norm(e["name"]))
        if real is None:
            invented.append(e["name"])
        elif real not in matched:          # keep the first answer for a duplicate name
            matched[real] = e
    return matched, invented


# metrics
def _ranks(values):
    """Average ranks (1 = smallest), ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation = Pearson correlation of the (tie-averaged) ranks."""
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None                     
    return cov / (vx * vy) ** 0.5


def compare(real, matched, invented):
    """Per-neighbourhood rows and the summary metrics for one model/mode."""
    rows = []
    for r in real.itertuples(index=False):
        e = matched.get(r.neighbourhood)
        row = {"neighbourhood": r.neighbourhood, "real_monthly_avg": round(r.real_monthly_avg, 2),
               "real_rank": r.real_rank, "llm_rating": None, "llm_estimate": None,
               "abs_pct_error": None, "status": "skipped"}
        if e is not None:
            row.update(llm_rating=e["rating"], llm_estimate=e["incidents_per_month"], status="matched")
            if r.real_monthly_avg > 0:
                row["abs_pct_error"] = round(abs(e["incidents_per_month"] - r.real_monthly_avg)
                                             / r.real_monthly_avg * 100, 1)
        rows.append(row)
    for name in invented:
        rows.append({"neighbourhood": name, "real_monthly_avg": None, "real_rank": None,
                     "llm_rating": None, "llm_estimate": None, "abs_pct_error": None,
                     "status": "invented"})

    m = [r for r in rows if r["status"] == "matched"]
    real_top5 = set(real.nsmallest(5, "real_rank")["neighbourhood"])
    # LLM top 5: highest rating, ties broken by its own incident estimate
    llm_top5 = {r["neighbourhood"] for r in
                sorted(m, key=lambda r: (r["llm_rating"], r["llm_estimate"]), reverse=True)[:5]}
    errs = [r["abs_pct_error"] for r in m if r["abs_pct_error"] is not None]
    metrics = {
        "neighbourhoods_real": len(real),
        "neighbourhoods_matched": len(m),
        "spearman_rating_vs_real": spearman([r["llm_rating"] for r in m],
                                            [r["real_monthly_avg"] for r in m]),
        "spearman_estimate_vs_real": spearman([r["llm_estimate"] for r in m],
                                              [r["real_monthly_avg"] for r in m]),
        "top5_overlap": len(real_top5 & llm_top5),
        "llm_top5": sorted(llm_top5),
        "real_top5": sorted(real_top5),
        "median_abs_pct_error": statistics.median(errs) if errs else None,
        "skipped": [r["neighbourhood"] for r in rows if r["status"] == "skipped"],
        "invented": list(invented),
    }
    return rows, metrics
