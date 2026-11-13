
from pathlib import Path

# Find the dataset folder
project_folder = Path(__file__).resolve().parent.parent

dataset_path = project_folder / "dataset" / "chest_xray"

print("Dataset location:", dataset_path)

print("\nFolders found:")

if dataset_path.exists():
    for item in dataset_path.iterdir():
        print(item.name)
else:
    print("Dataset folder does not exist!")


# Dataset folders
splits = ["train", "val", "test"]

# Image categories
categories = ["NORMAL", "PNEUMONIA"]

# Supported image formats
image_extensions = {".jpg", ".jpeg", ".png"}

print("\nPNEUMOSCAN DATASET INSPECTION")
print("=" * 45)

# Count images in each dataset folder
for split in splits:

    print(f"\n{split.upper()} DATASET")

    total_images = 0

    for category in categories:

        folder = dataset_path / split / category

        if not folder.exists():
            print(f"{category}: Folder not found!")
            continue

        images = [
            file for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in image_extensions
        ]

        count = len(images)

        total_images += count

        print(f"{category}: {count} images")

    print(f"Total {split} images: {total_images}")

print("\nDataset inspection completed!")
