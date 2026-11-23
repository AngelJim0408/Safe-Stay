"""Build the neighbourhood x year x month table (Task 1)."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import classify
import config


def build_summary(df: DataFrame) -> DataFrame:
    base = (df.withColumn("year", F.year("incident_ts"))
              .withColumn("month", F.month("incident_ts"))
              .withColumnRenamed(config.COL_NEIGHBOURHOOD, "neighbourhood"))
    keys = ["neighbourhood", "year", "month"]

    counts = base.groupBy(*keys).agg(
        F.count("*").alias("total_incidents"),
        F.sum((F.col("crime_group") == classify.PERSONS).cast("int")).alias("persons"),
        F.sum((F.col("crime_group") == classify.PROPERTY).cast("int")).alias("property"),
        F.sum((F.col("crime_group") == classify.OTHER).cast("int")).alias("other"),
    )

    # top 3 offence types per neighbourhood-month
    per_type = base.groupBy(*keys, config.COL_CATEGORY).count()
    w = Window.partitionBy(*keys).orderBy(F.desc("count"), F.col(config.COL_CATEGORY))
    top3 = (per_type.withColumn("rk", F.row_number().over(w)).filter("rk <= 3")
            .groupBy(*keys)
            .agg(F.concat_ws("; ", F.collect_list(config.COL_CATEGORY)).alias("top3_offences")))

    return counts.join(top3, keys, "left").orderBy(*keys)
