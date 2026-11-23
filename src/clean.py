"""Spark cleaning steps. Each takes a DataFrame and returns a DataFrame."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import config


def cast_types(df: DataFrame) -> DataFrame:
    return (df
            .withColumn("incident_ts", F.to_timestamp(F.col(config.COL_DATETIME)))
            .withColumn("lat", F.col(config.COL_LAT).cast("double"))
            .withColumn("lon", F.col(config.COL_LON).cast("double")))


def drop_duplicates(df: DataFrame) -> DataFrame:
    """One report = one incident_id. The API returns one row per offence, so an
    incident with several offences appears several times. Keep a single row per
    incident_id (the one with the lowest row_id, so the result is deterministic)."""
    from pyspark.sql.window import Window

    w = Window.partitionBy(config.COL_INCIDENT_ID).orderBy(
        F.col(config.COL_ROW_ID).cast("long").asc_nulls_last())
    return (df.withColumn("_rn", F.row_number().over(w))
              .filter(F.col("_rn") == 1).drop("_rn"))


def drop_missing_date(df: DataFrame) -> DataFrame:
    return df.filter(F.col("incident_ts").isNotNull())


def drop_missing_neighbourhood(df: DataFrame) -> DataFrame:
    n = F.col(config.COL_NEIGHBOURHOOD)
    return df.filter(n.isNotNull() & (F.trim(n) != ""))


def drop_invalid_coords(df: DataFrame) -> DataFrame:
    return df.filter(F.col("lat").isNotNull() & F.col("lon").isNotNull()
                     & (F.col("lat") != 0) & (F.col("lon") != 0))


def filter_in_city(df: DataFrame) -> DataFrame:
    return df.filter(F.col("lat").between(config.LAT_MIN, config.LAT_MAX)
                     & F.col("lon").between(config.LON_MIN, config.LON_MAX))


def filter_date_range(df: DataFrame) -> DataFrame:
    return df.filter(F.col("incident_ts").between(f"{config.START_DATE} 00:00:00",
                                                  f"{config.END_DATE} 23:59:59"))
