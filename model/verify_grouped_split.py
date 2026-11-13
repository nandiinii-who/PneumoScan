
from pathlib import Path
from collections import defaultdict
import csv

from prepare_grouped_dataset import get_group_id

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "model" / "dataset_split_grouped.csv"

group_splits = defaultdict(set)
image_counts = defaultdict(int)

with open(CSV_PATH, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        filename = Path(row["filepath"]).name
        category = row["label"]
        split = row["split"]

        group_id = get_group_id(filename, category)

        group_splits[(category, group_id)].add(split)
        image_counts[(category, split)] += 1

overlapping_groups = [
    group
    for group, splits in group_splits.items()
    if len(splits) > 1
]

print("\nGROUPED DATASET VERIFICATION")
print("----------------------------")

print("Overlapping groups:", len(overlapping_groups))

print("\nImage counts:")
for category in ["NORMAL", "PNEUMONIA"]:
    print(
        category,
        "- Train:", image_counts[(category, "train")],
        "- Validation:", image_counts[(category, "val")]
    )

if overlapping_groups:
    print("\nExamples of overlapping groups:")
    for group in overlapping_groups[:10]:
        print(group)
else:
    print("\nPASS: No overlapping groups detected.")
