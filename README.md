
# 🫁 PneumoScan — AI-Powered Pneumonia Detection

PneumoScan is a deep learning-based web application that analyzes
chest X-ray images and classifies them as NORMAL or PNEUMONIA
using a Convolutional Neural Network (CNN).

The project combines artificial intelligence, image processing,
a Flask backend, and an interactive web interface to demonstrate
the application of machine learning in healthcare.

## ✨ Features

- Chest X-ray image upload and preview
- PNG and JPEG image support
- CNN-based pneumonia classification
- Real-time AI predictions
- Interactive model score visualization
- Classification threshold of 75%
- Dynamic NORMAL and PNEUMONIA result cards
- Flask REST API integration
- Image validation and error handling
- Reset and re-analysis functionality
- Educational and medical safety disclaimers

## 🛠️ Technology Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Python
- Flask
- Flask-CORS

### Artificial Intelligence
- TensorFlow
- Keras
- Convolutional Neural Network (CNN)
- NumPy
- Pillow

### Development Tools
- Visual Studio Code
- Git and GitHub
- Python Virtual Environment

## 🧠 CNN Model Architecture

The model processes chest X-ray images resized to 224 × 224 pixels
with three color channels.

Architecture:

1. Input Layer — 224 × 224 × 3
2. Conv2D — 32 filters, ReLU activation
3. MaxPooling2D
4. Conv2D — 64 filters, ReLU activation
5. MaxPooling2D
6. Conv2D — 128 filters, ReLU activation
7. MaxPooling2D
8. Flatten
9. Dense — 128 neurons, ReLU activation
10. Dropout — 0.5
11. Dense — 1 neuron, Sigmoid activation

Total parameters: 11,169,089

### Lightweight Inference Model

The final trained CNN was exported as a lightweight
inference model by removing the training optimizer
state while preserving the model architecture and
trained weights.

| Model | File Size |
|-------|-----------|
| Original trained model | 134.08 MB |
| Lightweight inference model | 44.71 MB |

This reduces the model size by approximately 66.7%.

The lightweight model is included in this repository:

`model/pneumoscan_inference.keras`

The Flask backend loads this model for predictions.
No additional model training is required to run the
web application.


## 📊 Dataset

The project uses the Chest X-Ray Images (Pneumonia)
dataset available on Kaggle.

Dataset:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains two categories:

- NORMAL
- PNEUMONIA

A grouped training-validation split was created using
identifiers extracted from image filenames to reduce
potential patient overlap between training and validation.

Training images: 4,121
Validation images: 1,095
Test images: 624

## 📈 Model Performance

The final model was evaluated using a classification
threshold of 0.75.

| Metric | Result |
|--------|--------|
| Accuracy | 78.21% |
| Precision | 74.61% |
| Recall | 98.72% |
| Specificity | 44.02% |
| F1 Score | 84.99% |
| Balanced Accuracy | 71.37% |

### Confusion Matrix

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| Actual NORMAL | 103 | 131 |
| Actual PNEUMONIA | 5 | 385 |

The classification threshold was selected using validation
data to improve the balance between recall and specificity.

## ⚙️ How It Works

1. The user uploads a chest X-ray image.
2. The frontend displays an image preview.
3. The image is sent to the Flask backend.
4. The backend validates and preprocesses the image.
5. The CNN generates a pneumonia model score.
6. The classification threshold is applied.
7. The prediction is returned to the frontend.
8. The results card displays the classification and model score.

## 📂 Project Structure

PneumoniaDetection/
    frontend/
        index.html
        style.css
        script.js

    backend/
        app.py
        image_preprocessing.py

    model/
        cnn_model.py
        data_loader.py
        train_model.py
        prepare_grouped_dataset.py
        verify_grouped_split.py
        evaluate_grouped_model.py
        analyze_threshold.py
        pneumoscan_inference.keras

    dataset/
        chest_xray/

    requirements.txt
    README.md

## 🔌 API Endpoints

### GET /api/health

Checks whether the Flask backend is running.

### POST /api/upload

Accepts a chest X-ray image and returns:

- Prediction
- Pneumonia model score
- Image information
- Educational disclaimer

The local backend runs on port 5001.


## 🚀 Running the Project Locally

### 1. Clone the repository

Clone this repository using GitHub and open the
project folder in Visual Studio Code.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Flask backend

From the main project directory:

```bash
python backend/app.py
```

The backend runs at:

http://127.0.0.1:5001

### 5. Open the frontend

Open `frontend/index.html` in your web browser,
or use the VS Code Live Server extension.

### 6. Analyze a chest X-ray

- Upload a PNG or JPEG chest X-ray.
- Click Analyze X-ray.
- View the AI classification and model score.
- Use Reset X-ray to analyze another image.

### Dataset and Training

The dataset and original training checkpoints are
not included in this repository due to their size.

The lightweight inference model is included, so
the web application can run without downloading
the training dataset.

To retrain the model, download the Kaggle dataset,
place it inside `dataset/chest_xray/`, generate the
grouped dataset split, and run the training scripts.


## ⚠️ Limitations

- The model has limited specificity and can incorrectly
  classify normal X-rays as pneumonia.
- The dataset may not represent all patient populations
  or imaging conditions.
- Filename-based grouping is an approximation and does
  not guarantee verified patient-level separation.
- The test dataset was inspected during development,
  so the reported results are not an independent
  external validation.
- The model is designed for binary classification and
  cannot diagnose other chest conditions.
- The model score is not a calibrated probability of
  a patient having pneumonia.
- The application does not verify that every uploaded
  image is a valid chest X-ray.

## 🔮 Future Scope

- Improve generalization using transfer learning.
- Evaluate the model on independent external datasets.
- Improve specificity and score calibration.
- Explore deployment and secure image processing.

## 🎓 Project Purpose

PneumoScan was developed as an educational deep learning
project to explore CNN architecture, medical image
classification, backend integration, and full-stack
AI application development.

## ⚕️ Medical Disclaimer

PneumoScan is an experimental educational and research
project. It is not a medical device and must not be used
for clinical diagnosis or treatment decisions.

Always consult a qualified healthcare professional
for medical advice.
