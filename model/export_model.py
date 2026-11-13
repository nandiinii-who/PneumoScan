
from pathlib import Path
from keras.models import load_model, clone_model

# Project paths
MODEL_FOLDER = Path(__file__).resolve().parent

ORIGINAL_MODEL = (
    MODEL_FOLDER / "best_pneumoscan_grouped.keras"
)

INFERENCE_MODEL = (
    MODEL_FOLDER / "pneumoscan_inference.keras"
)

# Load the trained model without optimizer
print("Loading original trained model...")

original_model = load_model(
    str(ORIGINAL_MODEL),
    compile=False
)

# Create a fresh copy of the architecture
print("Creating lightweight inference model...")

inference_model = clone_model(original_model)

# Copy all trained weights
inference_model.set_weights(
    original_model.get_weights()
)

# Save the lightweight model
inference_model.save(str(INFERENCE_MODEL))

# Display file sizes
original_size = (
    ORIGINAL_MODEL.stat().st_size / 1_000_000
)

inference_size = (
    INFERENCE_MODEL.stat().st_size / 1_000_000
)

print("\nModel export completed!")

print(f"Original model: {original_size:.2f} MB")

print(f"Inference model: {inference_size:.2f} MB")

print(f"\nSaved to: {INFERENCE_MODEL}")
