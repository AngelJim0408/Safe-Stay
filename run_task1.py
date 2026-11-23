"""Task 1: ingest -> clean -> classify -> summarise -> outputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import config  # noqa: E402
import classify  # noqa: E402
import clean  # noqa: E402
import ingest  # noqa: E402
import summarise  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from spark_utils import apply_step, get_spark, save_log  # noqa: E402


def main(skip_download: bool = False) -> None:
    config.ensure_dirs()
    if not skip_download and not config.RAW_PARQUET.exists():
        ingest.build_raw_parquet()

    spark = get_spark()
    df = spark.read.parquet(str(config.RAW_PARQUET))
    log = [{"step": "raw", "before": df.count(), "after": df.count(), "removed": 0}]

    df = clean.cast_types(df)
    df = apply_step(df, "drop_duplicates", clean.drop_duplicates, log)
    df = apply_step(df, "drop_missing_date", clean.drop_missing_date, log)
    df = apply_step(df, "drop_missing_neighbourhood", clean.drop_missing_neighbourhood, log)
    df = apply_step(df, "drop_invalid_coords", clean.drop_invalid_coords, log)
    df = apply_step(df, "filter_in_city", clean.filter_in_city, log)
    df = apply_step(df, "filter_date_range", clean.filter_date_range, log)
    save_log(log)

    classify.list_distinct_offences(df).show(200, truncate=False)   # inspect before mapping
    df = classify.add_category(df)

    table = summarise.build_summary(df)
    table.toPandas().to_parquet(config.NEIGHBOURHOOD_MONTH_PARQUET, index=False)

    top3 = (df.groupBy(config.COL_CATEGORY).count().orderBy(F.desc("count")).limit(3).collect())
    ts = df.agg(F.date_format(F.min("incident_ts"), "yyyy-MM-dd HH:mm").alias("lo"),
                F.date_format(F.max("incident_ts"), "yyyy-MM-dd HH:mm").alias("hi")).first()
    n_hoods = df.select(config.COL_NEIGHBOURHOOD).distinct().count()

    lines = ["Task 1 summary", "=" * 14, "Row counts per cleaning step:"]
    lines += [f"  {s['step']}: {s['before']:,} -> {s['after']:,} (removed {s['removed']:,})" for s in log]
    lines += [f"Final rows: {df.count():,}", f"Neighbourhoods: {n_hoods}",
              f"Date range: {ts['lo']} to {ts['hi']}", "Top 3 offence types overall:"]
    lines += [f"  {r[config.COL_CATEGORY]}: {r['count']:,}" for r in top3]
    config.T1_SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    spark.stop()


if __name__ == "__main__":
    main()
