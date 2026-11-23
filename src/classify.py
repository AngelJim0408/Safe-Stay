"""Map the city's offence field to: persons / property / other.

Rules (documented in README). Based on the distinct category/subcategory values in
the cleaned 2022-2024 data. Loosely follows FBI NIBRS groupings:
  persons  = offences where a person is the target (violence, intimidation, sex offences, stalking)
  property = theft, burglary, vehicle theft, arson, vandalism, fraud-type offences
  other    = society/regulatory offences (drugs, weapons, prostitution...), administrative
             or non-crime reports (lost property, warrants, collisions), and anything unmapped.
Unmapped or NULL values fall back to "other".
"""
import config

PERSONS = "crimes against persons"
PROPERTY = "crimes against property"
OTHER = "other"

# Keys are lower-cased, trimmed incident_category values.
PERSONS_CATEGORIES = {
    "assault",
    "homicide",
    "rape",
    "robbery",                      # force against a person
    "sex offense",
    "human trafficking (a), commercial sex acts",
    "human trafficking (b), involuntary servitude",
    "human trafficking, commercial sex acts",
    "offences against the family and children",   # stalking, domestic/child cases
}

PROPERTY_CATEGORIES = {
    "arson",
    "burglary",
    "larceny theft",
    "motor vehicle theft",
    "motor vehicle theft?",         # data-entry variant
    "malicious mischief",
    "vandalism",
    "fraud",
    "forgery and counterfeiting",
    "embezzlement",
    "stolen property",
}

# (category, subcategory) overrides for mixed catch-all categories. Lower-cased.
SUBCATEGORY_OVERRIDES = {
    # Intimidation is a crime against a person, even when filed as disorderly conduct
    ("disorderly conduct", "intimidation"): PERSONS,
    ("other miscellaneous", "intimidation"): PERSONS,
    ("other miscellaneous", "kidnapping"): PERSONS,
    # Property-type offences filed under "Other Miscellaneous"
    ("other miscellaneous", "fraud"): PROPERTY,
    ("other miscellaneous", "extortion-blackmail"): PROPERTY,
    ("other miscellaneous", "bribery"): PROPERTY,
}

# Everything else -> OTHER, including: drug offences, weapons offences, prostitution,
# gambling, liquor laws, disorderly conduct (other subcategories), trespass, loitering,
# missing person, suicide, lost property, recovered vehicle, warrants, traffic collisions
# and violations, non-criminal, case closure, courtesy report, fire report,
# suspicious occ, miscellaneous investigation, "Other" and NULL.


def map_offence(category, subcategory=None) -> str:
    """Pure-python mapping (unit-testable)."""
    cat = (category or "").strip().lower()
    sub = (subcategory or "").strip().lower()
    if (cat, sub) in SUBCATEGORY_OVERRIDES:
        return SUBCATEGORY_OVERRIDES[(cat, sub)]
    if cat in PERSONS_CATEGORIES:
        return PERSONS
    if cat in PROPERTY_CATEGORIES:
        return PROPERTY
    return OTHER


def add_category(df):
    """Spark version of map_offence. Adds column `crime_group`."""
    from pyspark.sql import functions as F

    cat = F.lower(F.trim(F.col(config.COL_CATEGORY)))
    sub = F.lower(F.trim(F.col(config.COL_SUBCATEGORY)))

    expr = None
    for (c, s), label in SUBCATEGORY_OVERRIDES.items():
        cond = (cat == c) & (sub == s)
        expr = F.when(cond, label) if expr is None else expr.when(cond, label)
    expr = (expr
            .when(cat.isin(*PERSONS_CATEGORIES), PERSONS)
            .when(cat.isin(*PROPERTY_CATEGORIES), PROPERTY)
            .otherwise(OTHER))
    return df.withColumn("crime_group", expr)


def list_distinct_offences(df):
    from pyspark.sql import functions as F

    return (df.groupBy(config.COL_CATEGORY, config.COL_SUBCATEGORY).count()
              .orderBy(config.COL_CATEGORY, F.desc("count")))
