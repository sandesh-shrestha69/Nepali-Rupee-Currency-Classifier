// ═══════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════

const API_URL = window.location.origin;
const WS_URL = window.location.protocol === 'https:' ? 'wss://' + window.location.host + '/ws/detect' : 'ws://' + window.location.host + '/ws/detect';

let webcamStream = null;
let websocket = null;
let detectionInterval = null;
let isDetecting = false;

// ═══════════════════════════════════════════════════════
// DOM ELEMENTS
// ═══════════════════════════════════════════════════════

// Upload mode
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('uploadPreview');
const previewImage = document.getElementById('previewImage');
const predictBtn = document.getElementById('predictBtn');
const clearUploadBtn = document.getElementById('clearUpload');
const uploadResults = document.getElementById('uploadResults');

// Real-time mode
const webcam = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const hiddenCanvas = document.getElementById('hiddenCanvas');
const startRealtimeBtn = document.getElementById('startRealtime');
const stopRealtimeBtn = document.getElementById('stopRealtime');
const detectionStatus = document.getElementById('detectionStatus');
const realtimeResults = document.getElementById('realtimeResults');

// Shared
const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');

// ═══════════════════════════════════════════════════════
// MODE SWITCHING
// ═══════════════════════════════════════════════════════

document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Update active button
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update active mode
        const mode = btn.dataset.mode;
        document.querySelectorAll('.mode-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${mode}-mode`).classList.add('active');

        // Stop detection if switching away
        if (mode === 'upload') {
            stopRealtimeDetection();
        }

        hideError();
    });
});

// ═══════════════════════════════════════════════════════
// UPLOAD MODE
// ═══════════════════════════════════════════════════════

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImageUpload(file);
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImageUpload(file);
    }
});

function handleImageUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';
        predictBtn.style.display = 'block';
        uploadResults.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

clearUploadBtn.addEventListener('click', () => {
    uploadArea.style.display = 'block';
    previewContainer.style.display = 'none';
    predictBtn.style.display = 'none';
    uploadResults.style.display = 'none';
    fileInput.value = '';
});

predictBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (file) {
        await predictCurrency(file);
    }
});

async function predictCurrency(file) {
    try {
        showLoading();
        hideError();

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        displayUploadResults(result);
        
    } catch (error) {
        showError('Failed to analyze currency. Please try again.');
        console.error('Error:', error);
    } finally {
        hideLoading();
    }
}

function displayUploadResults(result) {
    document.getElementById('uploadCurrency').textContent = result.currency;
    
    const confidencePercent = (result.confidence * 100).toFixed(2);
    document.getElementById('uploadConfidenceValue').textContent = `${confidencePercent}%`;
    document.getElementById('uploadConfidenceFill').style.width = `${confidencePercent}%`;
    
    uploadResults.style.display = 'block';
}

// ═══════════════════════════════════════════════════════
// REAL-TIME DETECTION MODE
// ═══════════════════════════════════════════════════════

startRealtimeBtn.addEventListener('click', startRealtimeDetection);
stopRealtimeBtn.addEventListener('click', stopRealtimeDetection);

async function startRealtimeDetection() {
    try {
        // Start webcam
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        webcam.srcObject = webcamStream;

        // Wait for video to load
        await new Promise(resolve => {
            webcam.onloadedmetadata = resolve;
        });

        // Setup canvas
        overlay.width = webcam.videoWidth;
        overlay.height = webcam.videoHeight;
        hiddenCanvas.width = webcam.videoWidth;
        hiddenCanvas.height = webcam.videoHeight;

        // Connect WebSocket
        connectWebSocket();

        // Update UI
        startRealtimeBtn.style.display = 'none';
        stopRealtimeBtn.style.display = 'inline-block';
        realtimeResults.style.display = 'block';
        detectionStatus.querySelector('.status-indicator').classList.add('active');
        detectionStatus.querySelector('span').textContent = 'Detecting...';

        hideError();
        
    } catch (error) {
        showError('Unable to access camera. Please check permissions.');
        console.error('Camera error:', error);
    }
}

function stopRealtimeDetection() {
    // Stop webcam
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
        webcamStream = null;
    }

    // Close WebSocket
    if (websocket) {
        websocket.close();
        websocket = null;
    }

    // Stop detection loop
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }

    isDetecting = false;

    // Update UI
    startRealtimeBtn.style.display = 'inline-block';
    stopRealtimeBtn.style.display = 'none';
    realtimeResults.style.display = 'none';
    detectionStatus.querySelector('.status-indicator').classList.remove('active');
    detectionStatus.querySelector('span').textContent = 'Ready to detect';

    // Clear overlay
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function connectWebSocket() {
    websocket = new WebSocket(WS_URL);

    websocket.onopen = () => {
        console.log('✅ WebSocket connected');
        isDetecting = true;
        startDetectionLoop();
    };

    websocket.onmessage = (event) => {
        const result = JSON.parse(event.data);
        displayRealtimeResults(result);
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        showError('Connection error. Please refresh.');
    };

    websocket.onclose = () => {
        console.log('WebSocket disconnected');
        isDetecting = false;
    };
}

function startDetectionLoop() {
    // Send frame every 500ms
    detectionInterval = setInterval(() => {
        if (isDetecting && websocket && websocket.readyState === WebSocket.OPEN) {
            captureAndSendFrame();
        }
    }, 500);
}

function captureAndSendFrame() {
    // Draw current video frame to hidden canvas
    const ctx = hiddenCanvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

    // Convert to base64
    const imageData = hiddenCanvas.toDataURL('image/jpeg', 0.8);

    // Send to server
    websocket.send(JSON.stringify({
        image: imageData
    }));
}

// ═══════════════════════════════════════════════════════
// DISPLAY REAL-TIME RESULTS
// ═══════════════════════════════════════════════════════

function displayRealtimeResults(result) {
    console.log('REALTIME RESULT:', result);  // DEBUG: Log every result
    
    if (result.error) {
        console.error('Detection error:', result.error);
        return;
    }

    // Update detection status
    const statusIndicator = detectionStatus.querySelector('.status-indicator');
    const statusText = detectionStatus.querySelector('span');
    
    if (result.detected) {
        statusIndicator.style.background = '#00ff88';
        statusText.textContent = `Detecting: ${result.currency}`;
        
        // Update badge
        document.getElementById('realtimeCurrency').textContent = result.currency;
        const confidencePercent = (result.confidence * 100).toFixed(1);
        document.getElementById('realtimeConfidence').textContent = `${confidencePercent}%`;
        
        // Show badge
        realtimeResults.style.display = 'block';
        
        // Draw bounding box
        drawBoundingBox(result);
    } else {
        console.log('SCANNING MODE - hiding results, conf:', result.confidence);
        
        // FORCE CLEAR STALE RESULTS
        document.getElementById('realtimeCurrency').textContent = '';
        document.getElementById('realtimeConfidence').textContent = '';
        realtimeResults.style.display = 'none';
        
        statusIndicator.style.background = '#ffa500';
        statusText.textContent = `Scanning... (${(result.confidence * 100).toFixed(1)}% - need ${(result.threshold * 100)}%+)`;
        
        // Clear bounding box
        clearBoundingBox();
    }
}

function clearBoundingBox() {
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawBoundingBox(result) {
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Only draw if detected AND has bbox
    if (!result.detected || !result.bbox) {
        return;
    }

    const bbox = result.bbox;

    // Convert normalized coordinates to pixels
    const x = bbox.x * overlay.width;
    const y = bbox.y * overlay.height;
    const width = bbox.width * overlay.width;
    const height = bbox.height * overlay.height;

    // Determine color based on confidence
    let boxColor;
    if (result.confidence >= 0.95) {
        boxColor = '#00ff88';  // Bright green - very confident
    } else if (result.confidence >= 0.85) {
        boxColor = '#00cc66';  // Green - confident
    } else if (result.confidence >= 0.70) {
        boxColor = '#ffa500';  // Orange - medium confidence
    }

    // Draw main box with glow effect
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 4;
    ctx.shadowBlur = 15;
    ctx.shadowColor = boxColor;
    ctx.strokeRect(x, y, width, height);
    ctx.shadowBlur = 0;

    // Draw label background
    const label = `${result.currency} - ${(result.confidence * 100).toFixed(1)}%`;
    ctx.font = 'bold 24px Arial';
    const textMetrics = ctx.measureText(label);
    const textWidth = textMetrics.width;
    const textHeight = 30;
    
    const labelX = x;
    const labelY = y - textHeight - 10;
    
    // Draw label background with rounded corners
    ctx.fillStyle = boxColor;
    roundRect(ctx, labelX, labelY, textWidth + 20, textHeight, 8);
    ctx.fill();

    // Draw label text
    ctx.fillStyle = '#000000';
    ctx.font = 'bold 20px Arial';
    ctx.fillText(label, labelX + 10, labelY + 22);

    // Draw corner markers (stylish touch)
    const markerLength = 30;
    const markerWidth = 4;
    ctx.lineWidth = markerWidth;
    ctx.strokeStyle = boxColor;

    // Top-left corner
    ctx.beginPath();
    ctx.moveTo(x, y + markerLength);
    ctx.lineTo(x, y);
    ctx.lineTo(x + markerLength, y);
    ctx.stroke();

    // Top-right corner
    ctx.beginPath();
    ctx.moveTo(x + width - markerLength, y);
    ctx.lineTo(x + width, y);
    ctx.lineTo(x + width, y + markerLength);
    ctx.stroke();

    // Bottom-left corner
    ctx.beginPath();
    ctx.moveTo(x, y + height - markerLength);
    ctx.lineTo(x, y + height);
    ctx.lineTo(x + markerLength, y + height);
    ctx.stroke();

    // Bottom-right corner
    ctx.beginPath();
    ctx.moveTo(x + width - markerLength, y + height);
    ctx.lineTo(x + width, y + height);
    ctx.lineTo(x + width, y + height - markerLength);
    ctx.stroke();

    // Draw scanning line animation (optional, looks cool)
    drawScanLine(x, y, width, height, boxColor);
}

// Helper function for rounded rectangles
function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

// Animated scan line
let scanLinePos = 0;
function drawScanLine(x, y, width, height, color) {
    const ctx = overlay.getContext('2d');
    
    // Animate scan line from top to bottom
    scanLinePos = (scanLinePos + 5) % height;
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(x, y + scanLinePos);
    ctx.lineTo(x + width, y + scanLinePos);
    ctx.stroke();
    ctx.globalAlpha = 1.0;
}

// ═══════════════════════════════════════════════════════
// UI HELPERS
// ═══════════════════════════════════════════════════════

function showLoading() {
    loading.style.display = 'block';
}

function hideLoading() {
    loading.style.display = 'none';
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function hideError() {
    errorDiv.style.display = 'none';
}

