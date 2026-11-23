"""Spark session factory and the tiny row-count logging helper."""
import json
from typing import Callable

from pyspark.sql import DataFrame, SparkSession

import config


def get_spark(app_name: str = "safestay", driver_memory: str = "4g") -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.warehouse.dir", str(config.SPARK_WAREHOUSE))
        .getOrCreate()
    )


def apply_step(df: DataFrame, name: str, fn: Callable[[DataFrame], DataFrame],
               log: list) -> DataFrame:
    """Run one cleaning step and record rows before/after in `log`."""
    before = df.count()
    out = fn(df)
    after = out.count()
    log.append({"step": name, "before": before, "after": after, "removed": before - after})
    print(f"[{name}] {before:,} -> {after:,} (removed {before - after:,})")
    return out


def save_log(log: list) -> None:
    config.ensure_dirs()
    config.ROW_LOG_JSON.write_text(json.dumps(log, indent=2))
