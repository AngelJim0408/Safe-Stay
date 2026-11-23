LLM-assisted trip planner built on open city data. City: San Francisco

Required libraries: 
pyspark
pandas
pyarrow
requests
python-dotenv
tabulate

Copy .env.example to .env and fill in the values (never commit .env):

Layout of the project:
Project layout
File - Purpose 
config.py - every constant: paths, dataset ID, date range, columns, bounding box, models
run_task1.py, run_task2.py - entry points
src/ingest.py - paged Socrata download, retries, resume, raw Parquet
src/spark_utils.py - Spark session, row-count logging
src/clean.py - Spark cleaning steps
src/classify.py - offence → persons / property / other
src/summarise.py - neighbourhood × year × month table
src/outlook.py - Task 2 calculations and the three lists
src/llm.py - LLM client (temperature 0) with provider fallback
src/grounding.py - number check: every number in a summary must appear in the data sent
outputs - all deliverable files 


How to run TASKS 1 and TASK 2:

python -m src.ingest              # smoke test: prints the real column names
python run_task1.py               # download, clean, classify, summarise
python run_task2.py --month 7     # outlook + LLM summary (add --no-llm to skip the LLM)
python run_task2.py --month 1

