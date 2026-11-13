
from pathlib import Path
import csv
import random

# Locate project folders
project_folder = Path(__file__).resolve().parent.parent

train_folder = project_folder / "dataset" / "chest_xray" / "train"

output_file = project_folder / "model" / "dataset_split.csv"

# Set random seed for reproducibility
random.seed(42)

categories = ["NORMAL", "PNEUMONIA"]

train_rows = []
val_rows = []

for category in categories:

    folder = train_folder / category

    images = sorted([
        file for file in folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    # Shuffle images randomly
    random.shuffle(images)

    # Reserve 20% for validation
    val_count = int(len(images) * 0.20)

    validation_images = images[:val_count]
    training_images = images[val_count:]

    for image in training_images:
        train_rows.append([
            str(image.relative_to(project_folder)),
            category,
            "train"
        ])

    for image in validation_images:
        val_rows.append([
            str(image.relative_to(project_folder)),
            category,
            "val"
        ])

# Save image paths and labels to CSV
with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["filepath", "label", "split"])

    writer.writerows(train_rows)
    writer.writerows(val_rows)

print("Dataset split created successfully!")

print("Training images:", len(train_rows))
print("Validation images:", len(val_rows))

print("Saved to:", output_file)
