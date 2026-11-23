"""Check that every number in an LLM summary appears in the rows we sent."""
import re

NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*%?")
# Numbers that are not data: years and "top N"-style small ordinals handled by caller
IGNORABLE_YEARS = set(range(2000, 2100))


def extract_numbers(text: str) -> list:
    out = []
    for tok in NUM_RE.findall(text):
        pct = tok.endswith("%")
        try:
            out.append((float(tok.rstrip("%").replace(",", "")), pct))
        except ValueError:
            continue
    return out


def allowed_values(rows: list) -> set:
    """All numeric values in the rows, plus percent forms of fractions (0.12 -> 12)."""
    vals = set()
    for row in rows:
        for v in row.values():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            vals.add(float(v))
            if 0 <= abs(v) <= 1:
                vals.add(float(v) * 100)   # share 0.12 may be written as 12%
    return vals


def _matches(x: float, allowed: set, rel_tol: float, abs_tol: float) -> bool:
    return any(abs(x - a) <= max(abs_tol, rel_tol * abs(a)) for a in allowed)


def verify(summary: str, rows: list, extra_ok=(5, 3, 2022, 2023, 2024),
           rel_tol: float = 0.01, abs_tol: float = 0.06):
    """Return (ok, unmatched). Tolerance covers rounding (e.g. 1.234 -> 1.23)."""
    allowed = allowed_values(rows)
    extra = {float(x) for x in extra_ok} | {float(y) for y in IGNORABLE_YEARS if y in (2022, 2023, 2024)}
    # month day-numbers like "July 4" are rare; add 1..12 only if you decide to allow them.
    bad = []
    for x, _pct in extract_numbers(summary):
        if x in extra or _matches(x, allowed, rel_tol, abs_tol):
            continue
        bad.append(x)
    return (not bad, bad)
