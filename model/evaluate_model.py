
from pathlib import Path
import numpy as np
import tensorflow as tf

# Locate project and saved model
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_FOLDER / "model" / "best_pneumoscan.keras"

TEST_FOLDER = PROJECT_FOLDER / "dataset" / "chest_xray" / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Labels must match training
LABEL_MAP = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}


# Step 1: Collect test images
image_paths = []
true_labels = []

for category, label in LABEL_MAP.items():

    folder = TEST_FOLDER / category

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    for image_path in sorted(folder.iterdir()):

        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:

            image_paths.append(str(image_path))
            true_labels.append(label)

if not image_paths:
    raise ValueError("No test images were found.")


# Step 2: Preprocess images
def load_test_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# Step 3: Create test dataset
test_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths, true_labels)
)

test_dataset = test_dataset.map(
    load_test_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# Step 4: Load the trained model
print("Loading trained PneumoScan model...")

model = tf.keras.models.load_model(str(MODEL_PATH))

print("Model loaded successfully!")


# Step 5: Generate predictions
print("\nEvaluating on", len(image_paths), "test images...")

probabilities = model.predict(test_dataset, verbose=1).reshape(-1)

# Use 0.5 as the initial classification threshold
predicted_labels = (probabilities >= 0.5).astype(int)

# Inspect prediction scores for each actual class

true_labels = np.array(true_labels)

normal_scores = probabilities[true_labels == 0]
pneumonia_scores = probabilities[true_labels == 1]

print("\nPREDICTION SCORE ANALYSIS")

print("NORMAL images:")
print("Average pneumonia score:",
      round(float(np.mean(normal_scores)), 4))
print("Median pneumonia score:",
      round(float(np.median(normal_scores)), 4))

print("\nPNEUMONIA images:")
print("Average pneumonia score:",
      round(float(np.mean(pneumonia_scores)), 4))
print("Median pneumonia score:",
      round(float(np.median(pneumonia_scores)), 4))


true_labels = np.array(true_labels)


# Step 6: Calculate confusion matrix values
# PNEUMONIA = positive class (1)

TP = int(np.sum((true_labels == 1) & (predicted_labels == 1)))

TN = int(np.sum((true_labels == 0) & (predicted_labels == 0)))

FP = int(np.sum((true_labels == 0) & (predicted_labels == 1)))

FN = int(np.sum((true_labels == 1) & (predicted_labels == 0)))


# Step 7: Calculate evaluation metrics
def safe_divide(a, b):
    return a / b if b else 0.0


accuracy = safe_divide(TP + TN, TP + TN + FP + FN)

precision = safe_divide(TP, TP + FP)

recall = safe_divide(TP, TP + FN)

specificity = safe_divide(TN, TN + FP)

f1_score = safe_divide(
    2 * precision * recall,
    precision + recall
)


# Step 8: Display results
print("\n" + "=" * 45)
print("PNEUMOSCAN MODEL EVALUATION")
print("=" * 45)

print("Total test images:", len(true_labels))

print(f"\nAccuracy:    {accuracy * 100:.2f}%")
print(f"Precision:   {precision * 100:.2f}%")
print(f"Recall:      {recall * 100:.2f}%")
print(f"Specificity: {specificity * 100:.2f}%")
print(f"F1 Score:    {f1_score * 100:.2f}%")

print("\nCONFUSION MATRIX")
print("                 Pred NORMAL   Pred PNEUMONIA")
print(f"Actual NORMAL       {TN:5d}          {FP:5d}")
print(f"Actual PNEUMONIA    {FN:5d}          {TP:5d}")

print("\nEvaluation completed!")
