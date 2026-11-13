
from pathlib import Path
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Project paths
MODEL_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = MODEL_FOLDER.parent

# Load both models
print("Loading original model...")

original_model = load_model(
    str(MODEL_FOLDER / "best_pneumoscan_grouped.keras"),
    compile=False
)

print("Loading lightweight model...")

inference_model = load_model(
    str(MODEL_FOLDER / "pneumoscan_inference.keras"),
    compile=False
)

# Test dataset
TEST_FOLDER = (
    PROJECT_FOLDER / "dataset" / "chest_xray" / "test"
)

THRESHOLD = 0.75

total_tests = 0
passed_tests = 0

# Test 5 NORMAL and 5 PNEUMONIA images
for category in ["NORMAL", "PNEUMONIA"]:

    print(f"\nTesting {category} images")
    print("-" * 45)

    image_files = sorted(
        (TEST_FOLDER / category).glob("*.jpeg")
    )[:5]

    for image_path in image_files:

        # Preprocess image
        image = tf.io.read_file(str(image_path))

        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image = tf.image.resize(image, (224, 224))

        image = tf.cast(image, tf.float32) / 255.0

        image = tf.expand_dims(image, axis=0)

        # Compare predictions
        original_score = float(
            original_model.predict(image, verbose=0)[0][0]
        )

        inference_score = float(
            inference_model.predict(image, verbose=0)[0][0]
        )

        original_class = (
            "PNEUMONIA"
            if original_score >= THRESHOLD
            else "NORMAL"
        )

        inference_class = (
            "PNEUMONIA"
            if inference_score >= THRESHOLD
            else "NORMAL"
        )

        # Check whether predictions match
        scores_match = np.isclose(
            original_score,
            inference_score,
            atol=1e-6,
            rtol=0
        )

        classes_match = (
            original_class == inference_class
        )

        total_tests += 1

        if scores_match and classes_match:

            passed_tests += 1
            status = "PASS"

        else:

            status = "FAIL"

        print(f"\nImage: {image_path.name}")

        print(f"Original: {original_score:.8f}")

        print(f"Lightweight: {inference_score:.8f}")

        print(f"Classification: {inference_class}")

        print(f"Result: {status}")

# Final summary
print("\n" + "=" * 45)
print("MODEL EXPORT VERIFICATION")
print("=" * 45)

print(f"Total tests: {total_tests}")
print(f"Passed: {passed_tests}")
print(f"Failed: {total_tests - passed_tests}")

if total_tests == 10 and passed_tests == 10:

    print("\nSUCCESS!")
    print("Both models produce matching predictions.")

else:

    print("\nVerification incomplete or failed.")
    print("Check the results before switching models.")
