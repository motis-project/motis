import argparse
import os
import subprocess
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--motis", default="./motis", help="path to the motis executable"
)
parser.add_argument(
    "-c", "--config", default="config.yml", help="path to the motis config file"
)
parser.add_argument(
    "-q", "--queries", default="queries.txt", help="path to the queries file"
)
parser.add_argument(
    "-r", "--responses", default="responses.txt", help="path to the responses file"
)
parser.add_argument(
    "-n",
    "--num_queries",
    default=1000,
    help="number of queries to use for the calibration",
)
args = parser.parse_args()

if not os.path.isfile(args.queries):
    subprocess.run(
        "{} generate -n {}".format(args.motis, args.num_queries), shell=True, check=True
    )

if not os.path.isfile(args.responses):
    subprocess.run(
        "{} batch -q {} -r {}".format(args.motis, args.queries, args.responses),
        shell=True,
        check=True,
    )

data = []
with open(args.responses) as f:
    for l in f:
        data.append(json.loads(l))
df = pd.json_normalize(data)

df["interval_length"] = df.apply(
    lambda row: int(row.nextPageCursor.removeprefix("LATER|"))
    - int(row.previousPageCursor.removeprefix("EARLIER|")),
    axis=1,
)

print(df["interval_length"].describe)
