
from pathlib import Path
import csv
import tensorflow as tf

# Locate the main project folder
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# Locate the CSV file created in Step 13
CSV_FILE = PROJECT_FOLDER / "model" / "dataset_split_grouped.csv"

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Convert dataset labels into numbers
LABEL_MAP = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}


def read_csv_data(split):

    image_paths = []
    labels = []

    with open(CSV_FILE, "r", newline="") as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["split"] == split:

                image_path = PROJECT_FOLDER / row["filepath"]

                image_paths.append(str(image_path))

                labels.append(LABEL_MAP[row["label"]])

    return image_paths, labels


def load_image(image_path, label):

    # Read the image file
    image = tf.io.read_file(image_path)

    # Decode PNG or JPEG images
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    # Resize to 224 x 224
    image = tf.image.resize(image, IMAGE_SIZE)

    # Normalize pixel values to 0-1
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(split):

    image_paths, labels = read_csv_data(split)

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle only the training dataset
    if split == "train":
        dataset = dataset.shuffle(
            buffer_size=min(len(image_paths), 1000),
            seed=42
        )

    # Group images into batches
    dataset = dataset.batch(BATCH_SIZE)

    # Prepare the next batch in advance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":

    train_dataset = create_dataset("train")

    val_dataset = create_dataset("val")

    print("Training dataset loaded!")
    print("Validation dataset loaded!")

    # Inspect one batch of training images
    for images, labels in train_dataset.take(1):

        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Pixel range:",
              float(tf.reduce_min(images)),
              "to",
              float(tf.reduce_max(images)))

        print("Sample labels:", labels.numpy())
