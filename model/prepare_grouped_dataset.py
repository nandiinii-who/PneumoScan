
from pathlib import Path
from collections import defaultdict
import csv
import random
import re

# =====================================
# PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "chest_xray" / "train"

OUTPUT_FILE = BASE_DIR / "model" / "dataset_split_grouped.csv"

RANDOM_SEED = 42
VALIDATION_RATIO = 0.20

rng = random.Random(RANDOM_SEED)

categories = ["NORMAL", "PNEUMONIA"]

train_rows = []
val_rows = []


# =====================================
# IDENTIFY IMAGE GROUPS
# =====================================

def get_group_id(filename, category):

    stem = Path(filename).stem

    if category == "PNEUMONIA":

        match = re.match(
            r"^(person\d+)_",
            stem,
            re.IGNORECASE
        )

        if match:
            return match.group(1).lower()

    elif category == "NORMAL":

        match = re.match(
            r"^((?:NORMAL2-)?IM-\d+)-",
            stem,
            re.IGNORECASE
        )

        if match:
            return match.group(1).lower()

    # Unknown naming pattern:
    # use the filename as its own group
    return "unknown_" + stem.lower()


# =====================================
# CREATE GROUPED SPLIT
# =====================================

for category in categories:

    folder = TRAIN_DIR / category

    groups = defaultdict(list)

    for image in sorted(folder.iterdir()):

        if image.suffix.lower() not in (
            ".jpg", ".jpeg", ".png"
        ):
            continue

        group_id = get_group_id(
            image.name,
            category
        )

        groups[group_id].append(image)

    group_ids = list(groups.keys())

    rng.shuffle(group_ids)

    # Allocate approximately 20% of groups
    # to validation
    val_group_count = int(
        len(group_ids) * VALIDATION_RATIO
    )

    validation_groups = set(
        group_ids[:val_group_count]
    )

    for group_id, images in groups.items():

        split = (
            "val"
            if group_id in validation_groups
            else "train"
        )

        for image in images:

            row = [
                str(image.relative_to(BASE_DIR)),
                category,
                split
            ]

            if split == "val":
                val_rows.append(row)
            else:
                train_rows.append(row)

    print(
        category,
        "- Total groups:",
        len(groups),
        "- Validation groups:",
        len(validation_groups)
    )


# =====================================
# SAVE NEW SPLIT
# =====================================

with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "filepath",
        "label",
        "split"
    ])

    writer.writerows(train_rows)
    writer.writerows(val_rows)


print("\nGrouped dataset split created!")

print("Training images:", len(train_rows))
print("Validation images:", len(val_rows))

print("Saved to:", OUTPUT_FILE)
