
from pathlib import Path
import numpy as np
import tensorflow as tf

from data_loader import create_dataset


# Step 1: Locate the trained model
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_FOLDER / "model" / "best_pneumoscan_grouped.keras"
)


# Step 2: Load validation dataset
print("Loading grouped validation dataset...")

val_dataset = create_dataset("val")


# Step 3: Load trained model
print("Loading grouped PneumoScan model...")

model = tf.keras.models.load_model(str(MODEL_PATH))


# Step 4: Collect predictions and actual labels
print("Generating validation predictions...")

all_scores = []
all_labels = []

for images, labels in val_dataset:

    scores = model.predict(images, verbose=0)

    all_scores.extend(scores.reshape(-1))
    all_labels.extend(labels.numpy().reshape(-1))

probabilities = np.array(all_scores)
true_labels = np.array(all_labels)

print("Total validation images:", len(true_labels))


# Step 5: Safe division
def safe_divide(a, b):
    return a / b if b else 0.0


# Step 6: Test different thresholds
thresholds = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97,
    0.99
]

results = []

print("\n" + "=" * 80)
print("PNEUMOSCAN THRESHOLD ANALYSIS")
print("=" * 80)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'Specificity':<14}"
    f"{'Balanced Acc':<14}"
)

print("-" * 80)


for threshold in thresholds:

    predicted_labels = (
        probabilities >= threshold
    ).astype(int)

    TP = np.sum(
        (true_labels == 1) & (predicted_labels == 1)
    )

    TN = np.sum(
        (true_labels == 0) & (predicted_labels == 0)
    )

    FP = np.sum(
        (true_labels == 0) & (predicted_labels == 1)
    )

    FN = np.sum(
        (true_labels == 1) & (predicted_labels == 0)
    )

    accuracy = safe_divide(
        TP + TN, TP + TN + FP + FN
    )

    precision = safe_divide(TP, TP + FP)

    recall = safe_divide(TP, TP + FN)

    specificity = safe_divide(TN, TN + FP)

    balanced_accuracy = (
        recall + specificity
    ) / 2

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy
    })

    print(
        f"{threshold:<12.2f}"
        f"{accuracy * 100:<12.2f}"
        f"{precision * 100:<12.2f}"
        f"{recall * 100:<12.2f}"
        f"{specificity * 100:<14.2f}"
        f"{balanced_accuracy * 100:<14.2f}"
    )


# Step 7: Find the best balanced threshold
best_result = max(
    results,
    key=lambda result: result["balanced_accuracy"]
)

print("\n" + "=" * 80)
print("BEST VALIDATION THRESHOLD")
print("=" * 80)

print("Threshold:", best_result["threshold"])

print(
    f"Accuracy: "
    f"{best_result['accuracy'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{best_result['recall'] * 100:.2f}%"
)

print(
    f"Specificity: "
    f"{best_result['specificity'] * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{best_result['balanced_accuracy'] * 100:.2f}%"
)

print("\nThreshold analysis completed!")
