/**
 * ================================================================================
 * MUDRA DETECTION ENGINE - Client-Side
 * ================================================================================
 * Handles webcam capture, API communication, and real-time mudra detection
 * 
 * Features:
 * - Optimized frame capture (configurable FPS)
 * - Efficient base64 encoding
 * - Real-time statistics
 * - Error handling and recovery
 * - Session management
 * 
 * Author: Mudra Academy
 * Date: January 2026
 * ================================================================================
 */

// ============================================================
// CONFIGURATION
// ============================================================
const CONFIG = {
    API_URL: 'http://localhost:5000',
    CAPTURE_INTERVAL: 150, // ms between frame captures (6-7 FPS for API)
    CANVAS_SIZE: { width: 640, height: 480 },
    IMAGE_QUALITY: 0.7, // JPEG quality (0.0 - 1.0)
    DEBUG: false
};

// ============================================================
// STATE MANAGEMENT
// ============================================================
const state = {
    isRunning: false,
    isPaused: false,
    stream: null,
    captureInterval: null,
    
    // Statistics
    stats: {
        detectionCount: 0,
        uniqueMudras: new Set(),
        totalConfidence: 0,
        frameCount: 0,
        lastFrameTime: 0,
        fps: 0
    },
    
    // Current detection
    currentMudra: null,
    currentConfidence: 0,
    currentMethod: 'NONE',
    
    // Detected mudras history
    detectedMudras: []
};

// ============================================================
// DOM ELEMENTS
// ============================================================
const elements = {
    webcam: document.getElementById('webcam'),
    canvas: document.getElementById('canvas'),
    startBtn: document.getElementById('startBtn'),
    pauseBtn: document.getElementById('pauseBtn'),
    stopBtn: document.getElementById('stopBtn'),
    videoContainer: document.getElementById('videoContainer'),
    
    // Result displays
    detectedMudra: document.getElementById('detectedMudra'),
    confidence: document.getElementById('confidence'),
    detectionMethod: document.getElementById('detectionMethod'),
    statusMessage: document.getElementById('statusMessage'),
    
    // Stats
    detectionCount: document.getElementById('detectionCount'),
    uniqueMudras: document.getElementById('uniqueMudras'),
    fpsDisplay: document.getElementById('fpsDisplay'),
    accuracy: document.getElementById('accuracy'),
    
    // Mudra list
    detectedMudrasList: document.getElementById('detectedMudrasList')
};

// Get canvas context
const ctx = elements.canvas.getContext('2d');

// ============================================================
// WEBCAM INITIALIZATION
// ============================================================
async function initializeWebcam() {
    try {
        updateStatus('Requesting camera access...', 'info');
        
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: CONFIG.CANVAS_SIZE.width },
                height: { ideal: CONFIG.CANVAS_SIZE.height },
                facingMode: 'user'
            },
            audio: false
        });
        
        state.stream = stream;
        elements.webcam.srcObject = stream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            elements.webcam.onloadedmetadata = () => {
                elements.webcam.play();
                resolve();
            };
        });
        
        // Set canvas size to match video
        elements.canvas.width = elements.webcam.videoWidth;
        elements.canvas.height = elements.webcam.videoHeight;
        
        updateStatus('Camera ready! Starting detection...', 'success');
        return true;
        
    } catch (error) {
        console.error('Webcam initialization error:', error);
        updateStatus('Camera access denied. Please enable camera permissions.', 'error');
        return false;
    }
}

// ============================================================
// FRAME CAPTURE & API COMMUNICATION
// ============================================================
async function captureAndDetect() {
    if (!state.isRunning || state.isPaused) return;
    
    try {
        // Calculate FPS
        const now = performance.now();
        if (state.stats.lastFrameTime > 0) {
            const delta = now - state.stats.lastFrameTime;
            state.stats.fps = Math.round(1000 / delta);
        }
        state.stats.lastFrameTime = now;
        state.stats.frameCount++;
        
        // Update FPS display
        elements.fpsDisplay.textContent = `${state.stats.fps} FPS`;
        
        // Capture frame from video
        ctx.drawImage(elements.webcam, 0, 0, elements.canvas.width, elements.canvas.height);
        
        // Convert to base64 (optimized quality)
        const imageData = elements.canvas.toDataURL('image/jpeg', CONFIG.IMAGE_QUALITY);
        
        // Send to API
        const response = await fetch(`${CONFIG.API_URL}/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                include_landmarks: false // Set to true if you want landmark data
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (CONFIG.DEBUG) {
            console.log('Detection result:', result);
        }
        
        // Process detection result
        processDetectionResult(result);
        
    } catch (error) {
        console.error('Detection error:', error);
        if (error.message.includes('Failed to fetch')) {
            updateStatus('⚠️ Lost connection to AI server. Retrying...', 'error');
        }
    }
}

// ============================================================
// DETECTION RESULT PROCESSING
// ============================================================
function processDetectionResult(result) {
    if (!result.success) {
        updateDisplay('Error', 0, 'ERROR');
        return;
    }
    
    if (!result.hand_detected) {
        updateDisplay('No hand detected', 0, 'NONE');
        return;
    }
    
    const mudraName = result.mudra;
    const confidence = result.confidence;
    const method = result.method;
    
    // Update display
    updateDisplay(mudraName, confidence, method);
    
    // Update statistics if valid detection
    if (mudraName !== 'Unknown' && confidence > 0) {
        state.stats.detectionCount++;
        state.stats.uniqueMudras.add(mudraName);
        state.stats.totalConfidence += confidence;
        
        // Add to detected mudras list
        addDetectedMudra(mudraName, confidence, method);
        
        // Update stats display
        updateStatisticsDisplay();
    }
    
    // Store current detection
    state.currentMudra = mudraName;
    state.currentConfidence = confidence;
    state.currentMethod = method;
}

// ============================================================
// UI UPDATES
// ============================================================
function updateDisplay(mudraName, confidence, method) {
    // Update mudra name
    elements.detectedMudra.textContent = mudraName;
    
    // Update confidence
    if (confidence > 0) {
        const confidencePercent = (confidence * 100).toFixed(1);
        elements.confidence.textContent = `Confidence: ${confidencePercent}%`;
        
        // Color code by confidence
        if (confidence >= 0.8) {
            elements.detectedMudra.style.color = '#16a34a'; // Green
        } else if (confidence >= 0.6) {
            elements.detectedMudra.style.color = '#ca8a04'; // Yellow
        } else {
            elements.detectedMudra.style.color = '#dc2626'; // Red
        }
    } else {
        elements.confidence.textContent = mudraName === 'No hand detected' ? 'Show your hand to the camera' : '';
        elements.detectedMudra.style.color = '#ffffff';
    }
    
    // Update method
    const methodLabels = {
        'RULE': '⚡ Rule-Based Detection',
        'ML': '🤖 ML Detection',
        'NONE': 'ONE HAND MODE',
        'ERROR': 'ERROR'
    };
    elements.detectionMethod.textContent = methodLabels[method] || method;
}

function updateStatisticsDisplay() {
    elements.detectionCount.textContent = state.stats.detectionCount;
    elements.uniqueMudras.textContent = state.stats.uniqueMudras.size;
    
    const avgConfidence = state.stats.detectionCount > 0 
        ? (state.stats.totalConfidence / state.stats.detectionCount * 100).toFixed(1)
        : 0;
    elements.accuracy.textContent = `${avgConfidence}%`;
}

function addDetectedMudra(mudraName, confidence, method) {
    // Check if already in list
    const existing = state.detectedMudras.find(m => m.name === mudraName);
    
    if (existing) {
        existing.count++;
        existing.lastConfidence = confidence;
        existing.method = method;
    } else {
        state.detectedMudras.push({
            name: mudraName,
            count: 1,
            lastConfidence: confidence,
            method: method
        });
    }
    
    // Sort by count (descending)
    state.detectedMudras.sort((a, b) => b.count - a.count);
    
    // Update list display
    updateMudrasList();
}

function updateMudrasList() {
    if (state.detectedMudras.length === 0) {
        elements.detectedMudrasList.innerHTML = `
            <div style="text-align: center; color: var(--color-text-light); font-size: 0.9rem; padding: 1rem;">
                No mudras detected yet
            </div>
        `;
        return;
    }
    
    elements.detectedMudrasList.innerHTML = state.detectedMudras
        .slice(0, 10) // Show top 10
        .map(mudra => {
            const isCurrentlyDetected = mudra.name === state.currentMudra;
            const methodIcon = mudra.method === 'RULE' ? '⚡' : '🤖';
            const confidence = (mudra.lastConfidence * 100).toFixed(0);
            
            return `
                <div class="mudra-item ${isCurrentlyDetected ? 'detected' : ''}">
                    ${methodIcon} ${mudra.name}
                    <small style="float: right; opacity: 0.7;">
                        ${mudra.count}x • ${confidence}%
                    </small>
                </div>
            `;
        })
        .join('');
}

function updateStatus(message, type = 'info') {
    elements.statusMessage.textContent = message;
    elements.statusMessage.className = `status-message status-${type}`;
}

// ============================================================
// DETECTION CONTROL
// ============================================================
async function startDetection() {
    if (state.isRunning) return;
    
    // Disable start button
    elements.startBtn.disabled = true;
    elements.startBtn.innerHTML = '<span class="spinner"></span> Starting...';
    
    // Initialize webcam
    const webcamReady = await initializeWebcam();
    
    if (!webcamReady) {
        elements.startBtn.disabled = false;
        elements.startBtn.innerHTML = '🚀 Start Detection';
        return;
    }
    
    // Start detection loop
    state.isRunning = true;
    state.isPaused = false;
    
    // Add detecting animation
    elements.videoContainer.classList.add('detecting');
    
    // Start capture interval
    state.captureInterval = setInterval(captureAndDetect, CONFIG.CAPTURE_INTERVAL);
    
    // Update buttons
    elements.startBtn.disabled = true;
    elements.pauseBtn.disabled = false;
    elements.stopBtn.disabled = false;
    elements.startBtn.innerHTML = '✓ Running';
    
    updateStatus('Detection active! Show mudras to your camera.', 'success');
}

function togglePause() {
    state.isPaused = !state.isPaused;
    
    if (state.isPaused) {
        elements.pauseBtn.innerHTML = '▶️ Resume';
        elements.videoContainer.classList.remove('detecting');
        updateStatus('Detection paused', 'info');
    } else {
        elements.pauseBtn.innerHTML = '⏸️ Pause';
        elements.videoContainer.classList.add('detecting');
        updateStatus('Detection resumed', 'success');
    }
}

function stopDetection() {
    // Stop capture interval
    if (state.captureInterval) {
        clearInterval(state.captureInterval);
        state.captureInterval = null;
    }
    
    // Stop webcam
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }
    
    // Reset state
    state.isRunning = false;
    state.isPaused = false;
    
    // Remove detecting animation
    elements.videoContainer.classList.remove('detecting');
    
    // Update buttons
    elements.startBtn.disabled = false;
    elements.pauseBtn.disabled = true;
    elements.stopBtn.disabled = true;
    elements.startBtn.innerHTML = '🚀 Start Detection';
    elements.pauseBtn.innerHTML = '⏸️ Pause';
    
    // Clear video
    elements.webcam.srcObject = null;
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    
    // Reset display
    updateDisplay('Session Ended', 0, 'NONE');
    updateStatus(`Session completed! Detected ${state.stats.uniqueMudras.size} unique mudras.`, 'info');
}

// ============================================================
// INITIALIZATION
// ============================================================
async function checkAPIConnection() {
    try {
        const response = await fetch(`${CONFIG.API_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            updateStatus(`AI Server connected • ${data.mudra_count} mudras ready`, 'success');
            return true;
        }
    } catch (error) {
        updateStatus('⚠️ AI Server offline. Please start the server first.', 'error');
        elements.startBtn.disabled = true;
        return false;
    }
}

// Check API on page load
checkAPIConnection();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (state.isRunning) {
        stopDetection();
    }
});

// Make functions globally available
window.startDetection = startDetection;
window.togglePause = togglePause;
window.stopDetection = stopDetection;

console.log('Mudra Detection Engine initialized ✓');
