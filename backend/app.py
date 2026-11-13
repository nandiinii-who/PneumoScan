from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from image_preprocessing import preprocess_image
from pathlib import Path
from keras.models import load_model

import io


# =====================================
# LOAD TRAINED PNEUMOSCAN CNN MODEL
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_PATH = BASE_DIR / "model" / "pneumoscan_inference.keras"


print("Loading PneumoScan CNN model...")
print("Model path:", MODEL_PATH)


model = load_model(str(MODEL_PATH), compile=False)


print("PneumoScan CNN model loaded successfully!")


# Create Flask application
app = Flask(__name__)


# Allow frontend requests during local development
CORS(app, resources={
    r"/api/*": {"origins": "*"}
})


# Maximum request size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# Health check endpoint
@app.route("/api/health", methods=["GET"])
def health_check():

    return jsonify({
        "status": "success",
        "message": "PneumoScan backend is running!"
    })


# X-ray upload endpoint
@app.route("/api/upload", methods=["POST"])
def upload_xray():

    # Check whether a file was sent
    if "image" not in request.files:
        
        return jsonify({
            "status": "error",
            "message": "No image was uploaded."
        }), 400

    file = request.files["image"]

    # Check whether a file was selected
    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Please select an image."
        }), 400

    # Check file extension
    allowed_extensions = (".png", ".jpg", ".jpeg")

    if not file.filename.lower().endswith(allowed_extensions):
        return jsonify({
            "status": "error",
            "message": "Only PNG and JPEG images are allowed."
        }), 400

    # Verify actual image contents
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()

        image = Image.open(io.BytesIO(image_bytes))
        image.load()

        if image.format not in ("PNG", "JPEG"):
            raise ValueError("Unsupported image format")

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "status": "error",
            "message": "Invalid or corrupted image."
        }), 400


    # Preprocess the uploaded X-ray
    processed_image = preprocess_image(image)

    # =====================================
    # CNN PNEUMONIA PREDICTION
    # =====================================

    # Generate a prediction from the trained model
    prediction = model.predict(
        processed_image,
        verbose=0
    )

   
    # Extract pneumonia prediction score
    pneumonia_score = float(prediction[0][0])

    # Classification threshold selected from validation
    THRESHOLD = 0.75

    # Classify using the new threshold
    if pneumonia_score >= THRESHOLD:
        predicted_class = "PNEUMONIA"
    else:
        predicted_class = "NORMAL"

    print("Predicted class:", predicted_class)
    print("Pneumonia score:", pneumonia_score)
    print("Classification threshold:", THRESHOLD)

    print("Preprocessed image shape:", processed_image.shape)

    print("Minimum pixel value:", processed_image.min())

    print("Maximum pixel value:", processed_image.max())

    # Confirm successful upload
    
    return jsonify({
        "status": "success",
        "message": "X-ray analyzed successfully!",
        "filename": file.filename,
        "width": image.width,
        "height": image.height,
        "prediction": predicted_class,
        "pneumonia_score": round(pneumonia_score, 4),
        "disclaimer": (
            "Experimental educational AI output only. "
            "Not a medical diagnosis."
        )
    })



# Start Flask server
if __name__ == "__main__":
    app.run(debug=True, port=5001)
