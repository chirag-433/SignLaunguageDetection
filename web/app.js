/* ═══════════════════════════════════════════════════════════════════
   SignSpeak — Main Application Logic
   Real-time hand sign → text → speech, fully in the browser.
   ═══════════════════════════════════════════════════════════════════ */

import { FilesetResolver, HandLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// ── DOM Elements ────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const statusDot       = $("statusDot");
const statusText      = $("statusText");
const cameraRes       = $("cameraResolution");
const cameraPlaceholder = $("cameraPlaceholder");
const btnStart        = $("btnStart");
const video           = $("webcamVideo");
const canvas          = $("overlayCanvas");
const ctx             = canvas.getContext("2d");
const fpsCounter      = $("fpsCounter");
const detectedLetter  = $("detectedLetter");
const detectionBadge  = $("detectionBadge");
const stabilityValue  = $("stabilityValue");
const stabilityFill   = $("stabilityFill");
const sentenceDisplay = $("sentenceDisplay");
const charCount       = $("charCount");
const referenceGrid   = $("referenceGrid");
const toastContainer  = $("toastContainer");
const btnSpeak        = $("btnSpeak");
const btnSpace        = $("btnSpace");
const btnBackspace    = $("btnBackspace");
const btnClear        = $("btnClear");

// ── State ───────────────────────────────────────────────────────────
let handLandmarker = null;
let onnxSession    = null;
let labelsMap      = {};
let stream         = null;
let running        = false;

let sentence       = "";
let lastCharacter  = null;
let stableCount    = 0;
let letterAdded    = false;
let lastAddTime    = 0;

const STABLE_THRESHOLD = 15;
const ADD_COOLDOWN_MS  = 1000;

// FPS tracking
let frameCount = 0;
let fpsTime    = performance.now();

// ── MediaPipe Hand Connections (for drawing skeleton) ───────────────
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],       // thumb
  [0,5],[5,6],[6,7],[7,8],       // index
  [0,9],[9,10],[10,11],[11,12],  // middle  (5→9 via palm)
  [0,13],[13,14],[14,15],[15,16],// ring
  [0,17],[17,18],[18,19],[19,20],// pinky
  [5,9],[9,13],[13,17]           // palm cross-connections
];

// ── Initialization ──────────────────────────────────────────────────

/**
 * Load labels.json
 */
async function loadLabels() {
  try {
    const resp = await fetch("labels.json");
    labelsMap = await resp.json();
    buildReferenceGrid();
  } catch (e) {
    console.error("Failed to load labels.json:", e);
    toast("⚠ Could not load labels.json", "info");
  }
}

/**
 * Build the sign reference grid in the sidebar
 */
function buildReferenceGrid() {
  referenceGrid.innerHTML = "";
  const entries = Object.entries(labelsMap).sort((a, b) => Number(a[0]) - Number(b[0]));
  for (const [classIdx, letter] of entries) {
    const item = document.createElement("div");
    item.className = "reference-item";
    item.id = `ref-${classIdx}`;
    item.innerHTML = `
      <span class="reference-item__letter">${letter}</span>
      <span class="reference-item__class">cls ${classIdx}</span>
    `;
    referenceGrid.appendChild(item);
  }
}

/**
 * Initialize MediaPipe HandLandmarker
 */
async function initHandLandmarker() {
  setStatus("Loading MediaPipe...", "loading");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.3,
    minHandPresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
}

/**
 * Initialize ONNX Runtime session
 */
async function initOnnx() {
  setStatus("Loading ONNX model...", "loading");
  try {
    onnxSession = await ort.InferenceSession.create("model.onnx");
  } catch (e) {
    console.error("ONNX load error:", e);
    toast("⚠ Could not load model.onnx — run export_model_onnx.py first", "info");
    throw e;
  }
}

// ── Camera ──────────────────────────────────────────────────────────

async function startCamera() {
  btnStart.classList.add("loading");
  btnStart.disabled = true;

  try {
    // Initialize ML models in parallel with camera
    await Promise.all([
      initHandLandmarker(),
      initOnnx(),
    ]);

    setStatus("Starting camera...", "loading");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    // Set canvas size to match video
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    cameraRes.textContent = `${video.videoWidth}×${video.videoHeight}`;

    // Hide placeholder
    cameraPlaceholder.classList.add("hidden");
    setStatus("Active", "active");
    toast("✓ Camera started", "success");

    running = true;
    requestAnimationFrame(processFrame);

  } catch (err) {
    console.error("Start error:", err);
    setStatus("Error", "error");
    toast("❌ " + err.message, "info");
    btnStart.classList.remove("loading");
    btnStart.disabled = false;
  }
}

function stopCamera() {
  running = false;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  cameraPlaceholder.classList.remove("hidden");
  setStatus("Stopped", "idle");
  toast("Camera stopped", "info");
  btnStart.classList.remove("loading");
  btnStart.disabled = false;
}

// ── Frame Processing Loop ───────────────────────────────────────────

function processFrame() {
  if (!running) return;

  const now = performance.now();

  // FPS counter
  frameCount++;
  if (now - fpsTime >= 1000) {
    fpsCounter.textContent = `${frameCount} FPS`;
    frameCount = 0;
    fpsTime = now;
  }

  // Run hand detection
  const results = handLandmarker.detectForVideo(video, now);

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  let predictedCharacter = null;

  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0]; // first hand

    // Draw skeleton overlay
    drawHandSkeleton(landmarks);

    // Normalize landmarks (same math as Python version)
    const xs = landmarks.map((l) => l.x);
    const ys = landmarks.map((l) => l.y);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);

    const features = [];
    for (const lm of landmarks) {
      features.push(lm.x - minX);
      features.push(lm.y - minY);
    }

    // Run ONNX inference
    predictedCharacter = runInference(features);

    // Draw bounding box
    drawBoundingBox(xs, ys, minX, minY, predictedCharacter);
  }

  // Stability logic
  handleStability(predictedCharacter, now);

  requestAnimationFrame(processFrame);
}

// ── ONNX Inference ──────────────────────────────────────────────────

function runInference(features) {
  if (!onnxSession) return null;

  try {
    const inputTensor = new ort.Tensor("float32", Float32Array.from(features), [1, features.length]);
    const feeds = {};

    // Get the input name from the session
    const inputName = onnxSession.inputNames[0];
    feeds[inputName] = inputTensor;

    // Synchronous run via the session (we'll use runSync-like pattern)
    // onnxruntime-web's run is async, so we cache results
    // For real-time we use a fire-and-forget pattern with last-known result
    onnxSession.run(feeds).then((output) => {
      const labelTensor = output[onnxSession.outputNames[0]];
      const predictedClass = labelTensor.data[0];
      lastInferenceResult = labelsMap[String(predictedClass)] || null;
    }).catch(() => {});

    return lastInferenceResult;
  } catch {
    return null;
  }
}

let lastInferenceResult = null;

// ── Drawing Helpers ─────────────────────────────────────────────────

function drawHandSkeleton(landmarks) {
  const w = canvas.width;
  const h = canvas.height;

  // Colors for fingers
  const getDotColor = (i) => {
    if (i === 0) return "#ffffff"; // wrist
    if (i <= 4) return "#FF5252"; // thumb
    if (i <= 8) return "#FF9800"; // index
    if (i <= 12) return "#CDDC39"; // middle
    if (i <= 16) return "#00BCD4"; // ring
    return "#7C4DFF"; // pinky
  };

  // Draw connections
  ctx.lineWidth = 3;
  for (const [i, j] of HAND_CONNECTIONS) {
    const from = landmarks[i];
    const to   = landmarks[j];
    
    ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
    ctx.beginPath();
    ctx.moveTo((1 - from.x) * w, from.y * h);
    ctx.lineTo((1 - to.x) * w, to.y * h);
    ctx.stroke();
  }

  // Draw landmark dots
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const x = (1 - lm.x) * w; // mirrored
    const y = lm.y * h;

    // Outer border
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fill();

    // Inner dot
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = getDotColor(i);
    ctx.fill();
  }
}

function drawBoundingBox(xs, ys, minX, minY, character) {
  const w = canvas.width;
  const h = canvas.height;
  const pad = 20;

  // xs and minX are unmirrored. To draw mirrored, flip coordinates:
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);

  const x1 = (1 - maxX) * w - pad;
  const y1 = minY * h - pad;
  const boxW = (maxX - minX) * w + (pad * 2);
  const boxH = (maxY - minY) * h + (pad * 2);

  // Rounded rectangle
  ctx.strokeStyle = "rgba(0, 245, 212, 0.7)";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 4]);
  ctx.strokeRect(x1, y1, boxW, boxH);
  ctx.setLineDash([]);

  // Label
  if (character) {
    ctx.font = "bold 26px 'Inter', sans-serif";
    ctx.fillStyle = "#00f5d4";
    ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
    ctx.shadowBlur = 6;
    ctx.fillText(character, x1 + 4, y1 - 10);
    ctx.shadowBlur = 0;
  }
}

// ── Stability Engine ────────────────────────────────────────────────

function handleStability(character, now) {
  if (character !== null) {
    if (character === lastCharacter) {
      stableCount++;
    } else {
      stableCount = 1;
      letterAdded = false;
    }
    lastCharacter = character;

    // Update UI
    const progress = Math.min(stableCount / STABLE_THRESHOLD, 1.0);
    stabilityFill.style.width = `${progress * 100}%`;
    stabilityValue.textContent = `${Math.round(progress * 100)}%`;

    if (progress >= 1.0) {
      stabilityFill.classList.add("meter__fill--locked");
    } else {
      stabilityFill.classList.remove("meter__fill--locked");
    }

    // Update detected letter display
    updateDetectedLetter(character);
    setDetectionStatus("detecting");

    // Lock in letter
    if (stableCount >= STABLE_THRESHOLD && !letterAdded) {
      if (now - lastAddTime >= ADD_COOLDOWN_MS) {
        sentence += character;
        letterAdded = true;
        lastAddTime = now;
        updateSentenceDisplay();
        toast(`✓ Added '${character}'`, "success");

        // Pop animation on the big letter
        detectedLetter.classList.remove("detected-letter--pop");
        void detectedLetter.offsetWidth; // reflow trigger
        detectedLetter.classList.add("detected-letter--pop");
      }
    }
  } else {
    // No hand detected
    stableCount = 0;
    lastCharacter = null;
    letterAdded = false;

    stabilityFill.style.width = "0%";
    stabilityValue.textContent = "0%";
    stabilityFill.classList.remove("meter__fill--locked");

    updateDetectedLetter(null);
    setDetectionStatus("idle");
  }

  // Highlight active reference item
  highlightReference(character);
}

// ── UI Updates ──────────────────────────────────────────────────────

function updateDetectedLetter(letter) {
  if (letter) {
    detectedLetter.innerHTML = letter;
    detectedLetter.classList.remove("detected-letter__placeholder");
  } else {
    detectedLetter.innerHTML = `<span class="detected-letter__placeholder">—</span>`;
  }
}

function updateSentenceDisplay() {
  charCount.textContent = `${sentence.length} chars`;

  if (sentence.length === 0) {
    sentenceDisplay.classList.add("sentence-display--empty");
    sentenceDisplay.textContent = "Show a sign to start typing...";
    return;
  }

  sentenceDisplay.classList.remove("sentence-display--empty");

  // Build with individual letter spans for animation
  sentenceDisplay.innerHTML = "";
  for (let i = 0; i < sentence.length; i++) {
    const span = document.createElement("span");
    span.className = "letter-char";
    span.textContent = sentence[i] === " " ? "\u00A0" : sentence[i];
    // Only animate the last character
    if (i < sentence.length - 1) {
      span.style.animation = "none";
    }
    sentenceDisplay.appendChild(span);
  }
}

function setDetectionStatus(state) {
  statusDot.className = "status-dot";
  if (state === "detecting") {
    statusDot.classList.add("status-dot--detecting");
    detectionBadge.textContent = "Detecting";
  } else if (state === "active") {
    statusDot.classList.add("status-dot--active");
    detectionBadge.textContent = "Active";
  } else {
    detectionBadge.textContent = "Waiting";
  }
}

function setStatus(text, state) {
  statusText.textContent = text;
  statusDot.className = "status-dot";
  if (state === "active") statusDot.classList.add("status-dot--active");
  else if (state === "detecting") statusDot.classList.add("status-dot--detecting");
}

function highlightReference(letter) {
  // Remove all active states
  document.querySelectorAll(".reference-item--active").forEach((el) =>
    el.classList.remove("reference-item--active")
  );

  if (!letter) return;

  // Find the matching reference item
  for (const [classIdx, lbl] of Object.entries(labelsMap)) {
    if (lbl === letter) {
      const refEl = document.getElementById(`ref-${classIdx}`);
      if (refEl) refEl.classList.add("reference-item--active");
    }
  }
}

// ── Toast Notifications ─────────────────────────────────────────────

function toast(message, type = "info") {
  const el = document.createElement("div");
  el.className = `toast toast--${type}`;
  el.textContent = message;
  toastContainer.appendChild(el);

  setTimeout(() => {
    el.classList.add("removing");
    el.addEventListener("animationend", () => el.remove());
  }, 2500);
}

// ── Text-to-Speech ──────────────────────────────────────────────────

function speakSentence() {
  const text = sentence.trim();
  if (!text) {
    toast("Nothing to speak", "info");
    return;
  }

  if ("speechSynthesis" in window) {
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 0.9;
    utter.pitch = 1;
    window.speechSynthesis.speak(utter);
    toast(`🔊 Speaking: "${text}"`, "success");
  } else {
    toast("⚠ Speech synthesis not supported in this browser", "info");
  }
}

// ── Sentence Actions ────────────────────────────────────────────────

function addSpace() {
  sentence += " ";
  updateSentenceDisplay();
}

function deleteLast() {
  if (sentence.length > 0) {
    sentence = sentence.slice(0, -1);
    updateSentenceDisplay();
  }
}

function clearSentence() {
  sentence = "";
  updateSentenceDisplay();
  toast("Cleared", "info");
}

// ── Event Listeners ─────────────────────────────────────────────────

btnStart.addEventListener("click", () => {
  if (running) {
    stopCamera();
  } else {
    startCamera();
  }
});

btnSpeak.addEventListener("click", speakSentence);
btnSpace.addEventListener("click", addSpace);
btnBackspace.addEventListener("click", deleteLast);
btnClear.addEventListener("click", clearSentence);

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  // Don't capture if user is in an input field
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

  switch (e.key) {
    case " ":
      e.preventDefault();
      addSpace();
      break;
    case "Backspace":
      e.preventDefault();
      deleteLast();
      break;
    case "Enter":
      e.preventDefault();
      speakSentence();
      break;
    case "c":
    case "C":
      clearSentence();
      break;
    case "q":
    case "Q":
      if (running) stopCamera();
      break;
  }
});

// Button ripple effect
document.querySelectorAll(".btn").forEach((btn) => {
  btn.addEventListener("mousemove", (e) => {
    const rect = btn.getBoundingClientRect();
    btn.style.setProperty("--ripple-x", `${((e.clientX - rect.left) / rect.width) * 100}%`);
    btn.style.setProperty("--ripple-y", `${((e.clientY - rect.top) / rect.height) * 100}%`);
  });
});

// ── Boot ────────────────────────────────────────────────────────────
loadLabels();
