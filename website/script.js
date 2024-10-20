const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const arucoIdElement = document.getElementById('aruco-id');
const orderIdElement = document.getElementById('order-id');
let currentOrderId = null;

// Initialize webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        webcamElement.srcObject = stream;
    })
    .catch((err) => {
        console.error('Error accessing webcam:', err);
    });

// Function to capture frame and send to server
function captureFrameAndDetectAruco() {
    const canvas = canvasElement.getContext('2d');
    canvasElement.width = webcamElement.videoWidth;
    canvasElement.height = webcamElement.videoHeight;

    // Draw the current video frame to the canvas
    canvas.drawImage(webcamElement, 0, 0, canvasElement.width, canvasElement.height);

    // Show captured frame for debugging
    document.getElementById('debug-image').src = canvasElement.toDataURL('image/jpeg');

    // Get the image data URL (base64 format)
    const imageData = canvasElement.toDataURL('image/jpeg');

    // Send frame to the server for Aruco detection
    fetch('/detect_aruco', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        const detectedId = data.aruco_id;
        if (detectedId !== null) {
            arucoIdElement.textContent = detectedId;
            assignOrderId(detectedId);
        } else {
            arucoIdElement.textContent = 'No ID detected';
        }
    })
    .catch(err => console.error('Error detecting Aruco ID:', err));
}


// Assign order ID based on Aruco ID
function assignOrderId(arucoId) {
    currentOrderId = arucoId;
    orderIdElement.textContent = `Order #${currentOrderId}`;

    // After detection, start the model inferences
    runModels(currentOrderId);
}

// Run the grocery detection, fruit freshness detection, and OCR models
function runModels(orderId) {
    groceryDetection(orderId);
    fruitFreshnessDetection(orderId);
    ocrDetection(orderId);
}

// Grocery detection model (Mockup)
function groceryDetection(orderId) {
    document.getElementById('grocery-result').textContent = `Detected groceries for Order #${orderId}`;
}

// Fruit freshness detection model
function fruitFreshnessDetection(orderId) {
    fetch('inference_freshness.py', { method: 'POST', body: JSON.stringify({ orderId }) })
        .then(response => response.text())
        .then(data => {
            document.getElementById('freshness-result').textContent = `Freshness result for Order #${orderId}: ${data}`;
        })
        .catch(err => console.error('Error with freshness detection:', err));
}

// OCR model (Mockup)
function ocrDetection(orderId) {
    document.getElementById('ocr-result').textContent = `OCR result for Order #${orderId}`;
}

// Capture a frame and check for Aruco ID every time a frame is rendered
setInterval(captureFrameAndDetectAruco, 1000);  // Adjust interval as needed
