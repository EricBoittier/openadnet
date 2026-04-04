import csv
from pathlib import Path

data_path = Path(__file__).parent.parent / "src" / "data" / "kinase_inhibitors_after_2022.csv"

print(data_path)

with open(data_path, newline="") as f:
    reader = csv.DictReader(f)
    data = list(reader)

print(f"{len(data)} rows, columns: {list(data[0].keys())}")
for row in data[:5]:
    print(row)
