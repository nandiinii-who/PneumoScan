
from pathlib import Path
import csv
import re

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "model" / "dataset_split.csv"

train_patients = set()
val_patients = set()
unrecognized = []

with open(CSV_PATH, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        filename = Path(row["filepath"]).name

        match = re.match(r"^(person\d+)_", filename)

        if not match:
            unrecognized.append(filename)
            continue

        # Include the class to avoid mixing
        # potentially unrelated patient numbering
        patient_id = (
            row["label"],
            match.group(1)
        )

        if row["split"] == "train":
            train_patients.add(patient_id)

        elif row["split"] == "val":
            val_patients.add(patient_id)

overlap = train_patients.intersection(val_patients)

print("\nPATIENT OVERLAP CHECK")
print("-------------------------")
print("Training patient IDs:", len(train_patients))
print("Validation patient IDs:", len(val_patients))
print("Overlapping patient IDs:", len(overlap))
print("Unrecognized filenames:", len(unrecognized))

print("\nExamples of overlapping IDs:")
for patient in sorted(overlap)[:10]:
    print(patient)

print("\nExamples of unrecognized filenames:")
for filename in unrecognized[:10]:
    print(filename)
