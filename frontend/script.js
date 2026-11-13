
const xrayUpload = document.getElementById("xrayUpload");

const previewSection = document.getElementById("previewSection");

const xrayPreview = document.getElementById("xrayPreview");

const uploadStatus = document.getElementById("uploadStatus");

const resultMessage = document.getElementById("resultMessage");

const analyzeButton = document.getElementById("analyzeButton");

const resetButton = document.getElementById("resetButton");

const backendStatus = document.getElementById("backendStatus");

// New analysis result card elements

const resultCard =
    document.getElementById("resultCard");

const predictionValue =
    document.getElementById("predictionValue");

const predictionDescription =
    document.getElementById("predictionDescription");

const scoreValue =
    document.getElementById("scoreValue");

const scoreBar =
    document.getElementById("scoreBar");


let currentPreviewURL = null;

let uploadVersion = 0;

// =====================================
// CLEAR ANALYSIS RESULT CARD
// =====================================

function clearResultCard() {

    // Hide the previous result
    resultCard.style.display = "none";

    // Show the status message
    resultMessage.style.display = "block";

    // Reset prediction
    predictionValue.textContent = "";
    predictionValue.className = "";

    // Reset description
    predictionDescription.textContent = "";

    // Reset score
    scoreValue.textContent = "0%";

    // Reset progress bar
    scoreBar.style.width = "0%";
    scoreBar.className = "";

}



// =====================================
// 1. X-RAY IMAGE UPLOAD AND PREVIEW
// =====================================

xrayUpload.addEventListener("change", function () {

    uploadVersion++;

    // Clear previous analysis result
    clearResultCard();
    const file = xrayUpload.files[0];

    // Clear previous preview URL
    if (currentPreviewURL !== null) {
        URL.revokeObjectURL(currentPreviewURL);
        currentPreviewURL = null;
    }

    if (!file) {
        previewSection.style.display = "none";
        xrayPreview.removeAttribute("src");
        uploadStatus.textContent = "";
        analyzeButton.disabled = true;

        resultMessage.textContent =
            "No X-ray has been analyzed yet.";

        return;
    }

    // Validate image type
    if (
        file.type !== "image/png" &&
        file.type !== "image/jpeg"
    ) {

        alert("Please select a PNG or JPEG image.");

        xrayUpload.value = "";
        xrayPreview.removeAttribute("src");
        previewSection.style.display = "none";
        analyzeButton.disabled = true;

        resultMessage.textContent =
            "No X-ray has been analyzed yet.";

        return;
    }

    // Create image preview
    currentPreviewURL = URL.createObjectURL(file);

    xrayPreview.src = currentPreviewURL;

    previewSection.style.display = "block";

    analyzeButton.disabled = false;

    uploadStatus.textContent =
        "Chest X-ray selected successfully!";

    resultMessage.textContent =
        "Image uploaded. Analysis not started yet.";

});


// =====================================
// 2. RESET X-RAY
// =====================================

resetButton.addEventListener("click", function () {

    // Invalidate any previous upload request
    uploadVersion++;

    // Clear analysis result card
    clearResultCard();
    // Clear selected file
    xrayUpload.value = "";

    // Remove image
    xrayPreview.removeAttribute("src");

    // Hide preview
    previewSection.style.display = "none";

    // Clear status
    uploadStatus.textContent = "";

    // Reset result
    resultMessage.textContent =
        "No X-ray has been analyzed yet.";

    // Disable Analyze button
    analyzeButton.disabled = true;

    // Release image URL
    if (currentPreviewURL !== null) {
        URL.revokeObjectURL(currentPreviewURL);
        currentPreviewURL = null;
    }

});


// =====================================
// 3. FLASK BACKEND CONNECTION
// =====================================

async function checkBackendConnection() {

    backendStatus.textContent =
        "Checking backend connection...";

    backendStatus.className = "checking";

    try {

        const response = await fetch(
            "http://127.0.0.1:5001/api/health"
        );

        if (!response.ok) {
            throw new Error("Backend request failed");
        }

        const data = await response.json();

        if (data.status === "success") {

            backendStatus.textContent =
                "Backend Connected Successfully!";

            backendStatus.className = "connected";

        } else {

            throw new Error("Unexpected backend response");

        }

    } catch (error) {

        backendStatus.textContent =
            "Backend Disconnected. Check Flask server.";

        backendStatus.className = "disconnected";

        console.error("Backend Error:", error);

    }

}

// Check backend when page loads
checkBackendConnection();


// =====================================
// 4. SEND X-RAY TO FLASK BACKEND
// =====================================

analyzeButton.addEventListener("click", async function () {

    const file = xrayUpload.files[0];

    // Check if image exists
    if (!file) {

        alert("Please select a chest X-ray first.");

        return;
    }

    // Remember which image is being uploaded
    const requestVersion = uploadVersion;

    // Prepare image data
    const formData = new FormData();

    formData.append("image", file);

// Disable button during upload
analyzeButton.disabled = true;

// Clear previous analysis before starting
clearResultCard();

// Show analysis status
resultMessage.textContent =
    "Analyzing chest X-ray using AI...";


    try {

        // Send image to Flask
        const response = await fetch(
            "http://127.0.0.1:5001/api/upload",
            {
                method: "POST",
                body: formData
            }
        );

        // Read backend response
        const data = await response.json();

        // Ignore outdated responses
        if (requestVersion !== uploadVersion) {
            return;
        }

        // Check upload result
        if (!response.ok || data.status !== "success") {

            throw new Error(
                data.message || "Image upload failed."
            );

        }

        // Display successful upload
        
        // Display CNN prediction result
        const prediction = data.prediction;

        const pneumoniaScore = Number(data.pneumonia_score);

        // Verify that the backend returned valid prediction data
        if (
            !["NORMAL", "PNEUMONIA"].includes(prediction) ||
            !Number.isFinite(pneumoniaScore) ||
            pneumoniaScore < 0 ||
            pneumoniaScore > 1
        ) {
            throw new Error("Invalid prediction received from backend.");
        }

        
// Format prediction score
const scorePercentage =
    (pneumoniaScore * 100).toFixed(2);

// Clear previous analysis
clearResultCard();

// Display AI prediction
predictionValue.textContent = prediction;

// Display pneumonia model score
scoreValue.textContent = scorePercentage + "%";

// Set prediction colors and description
if (prediction === "PNEUMONIA") {

    predictionValue.className = "pneumonia";

    scoreBar.className = "pneumonia";

    predictionDescription.textContent =
        "The model detected patterns associated with pneumonia. " +
        "The model score meets or exceeds the 75% classification threshold.";

} else {

    predictionValue.className = "normal";

    scoreBar.className = "normal";

    predictionDescription.textContent =
        "The model classified this image as NORMAL. " +
        "The model score is below the 75% classification threshold. " +
        "This does not rule out pneumonia.";

}

// Hide the old text result
resultMessage.style.display = "none";

// Show the new analysis card
resultCard.style.display = "block";

// Start progress bar animation
scoreBar.style.width = "0%";

// Animate the progress bar
requestAnimationFrame(() => {

    requestAnimationFrame(() => {

        if (requestVersion === uploadVersion) {

            scoreBar.style.width =
                scorePercentage + "%";

        }

    });

});



    } catch (error) {

        // Ignore errors from old requests
        if (requestVersion !== uploadVersion) {
            return;
        }

        resultMessage.textContent =
            "Upload failed: " + error.message;

        console.error("Upload Error:", error);

    } finally {

        // Enable button only for the current image
        if (requestVersion === uploadVersion) {
            analyzeButton.disabled = false;
        }

    }

});
