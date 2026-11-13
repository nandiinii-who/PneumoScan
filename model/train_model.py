
import tensorflow as tf

# Import our CNN architecture
from cnn_model import build_cnn_model

# Import our dataset loader
from data_loader import create_dataset


# Step 1: Load the datasets
print("Loading training dataset...")

train_dataset = create_dataset("train")

print("Loading validation dataset...")

val_dataset = create_dataset("val")


# Step 2: Build our CNN model
print("Building CNN model...")

model = build_cnn_model()


# Step 3: Compile the model
model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]

)


# Step 4: Display model summary
model.summary()

print("\nCNN model compiled successfully!")
print("Training and validation datasets are ready.")
print("Model is ready for training.")


# Step 5: Calculate class weights automatically

import csv
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_FOLDER / "model" / "dataset_split_grouped.csv"
)

normal_count = 0
pneumonia_count = 0

# Count only training images
with open(CSV_PATH, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:

        if row["split"] != "train":
            continue

        if row["label"] == "NORMAL":
            normal_count += 1

        elif row["label"] == "PNEUMONIA":
            pneumonia_count += 1

# Ensure both classes are present
if normal_count == 0 or pneumonia_count == 0:
    raise ValueError(
        "Training split must contain both classes."
    )

total_count = normal_count + pneumonia_count

class_weights = {
    0: total_count / (2 * normal_count),
    1: total_count / (2 * pneumonia_count)
}

print("\nGROUPED TRAINING CLASS COUNTS")
print("NORMAL:", normal_count)
print("PNEUMONIA:", pneumonia_count)

print("\nCLASS WEIGHTS")
print(class_weights)


# Step 6: Configure training callbacks

from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# Save the new grouped model separately
MODEL_PATH = PROJECT_FOLDER / "model" / "best_pneumoscan_grouped.keras"

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

]



# Step 7: Train the CNN

print("\nStarting CNN training...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    class_weight=class_weights,
    callbacks=callbacks
)


# Step 8: Save training history

import json

history_path = PROJECT_FOLDER / "model" / "training_history_grouped.json"

with open(history_path, "w") as file:
    json.dump(history.history, file, indent=4)


print("\nTraining completed!")
print("Best model saved to:", MODEL_PATH)
print("Training history saved to:", history_path)
