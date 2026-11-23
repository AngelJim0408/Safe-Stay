"""Task 2 pandas calculations. Reads only the Task 1 Parquet (no Spark)."""
import pandas as pd

import config


def load_table() -> pd.DataFrame:
    return pd.read_parquet(config.NEIGHBOURHOOD_MONTH_PARQUET)


def complete_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing neighbourhood-year-month rows with 0 incidents."""
    idx = pd.MultiIndex.from_product(
        [df["neighbourhood"].unique(), config.YEARS, range(1, 13)],
        names=["neighbourhood", "year", "month"])
    out = df.set_index(["neighbourhood", "year", "month"]).reindex(idx)
    for c in ("total_incidents", "persons", "property", "other"):
        out[c] = out[c].fillna(0)
    return out.reset_index()


def compute_outlook(df: pd.DataFrame, month: int) -> pd.DataFrame:
    full = complete_grid(df)
    n_years = len(config.YEARS)
    m = full[full["month"] == month]
    level = m.groupby("neighbourhood").agg(
        avg_incidents=("total_incidents", lambda s: s.sum() / n_years),
        persons=("persons", lambda s: s.sum() / n_years),
        property=("property", lambda s: s.sum() / n_years)).reset_index()
    level["share_of_city"] = level["avg_incidents"] / level["avg_incidents"].sum()

    if config.SEASONAL_BASELINE == "all_months":
        base = full.groupby("neighbourhood")["total_incidents"].sum() / (12 * n_years)
    else:  # same 12 calendar months averaged -> identical formula; kept explicit
        base = full.groupby(["neighbourhood", "month"])["total_incidents"].mean() \
                   .groupby("neighbourhood").mean()
    level = level.merge(base.rename("baseline_avg"), on="neighbourhood")
    level["seasonal_ratio"] = level["avg_incidents"] / level["baseline_avg"]
    return level


def select_rows(level: pd.DataFrame) -> pd.DataFrame:
    """Top5 by volume, top5 by ratio (min-volume cutoff), bottom5 by count; dedupe."""
    n = config.TOP_N
    top_vol = level.nlargest(n, "avg_incidents").assign(list_name="most_incidents")
    eligible = level[level["avg_incidents"] >= config.MIN_MONTHLY_AVG_FOR_RATIO]
    top_ratio = eligible.nlargest(n, "seasonal_ratio").assign(list_name="largest_seasonal_increase")
    fewest = level.nsmallest(n, "avg_incidents").assign(list_name="fewest_incidents")
    rows = pd.concat([top_vol, top_ratio, fewest], ignore_index=True)
    # Overlaps: one row per neighbourhood, with every list it belongs to.
    merged = (rows.groupby("neighbourhood", sort=False)
                  .agg({**{c: "first" for c in rows.columns
                           if c not in ("neighbourhood", "list_name")},
                        "list_name": lambda s: "; ".join(s)})
                  .reset_index())
    return merged
