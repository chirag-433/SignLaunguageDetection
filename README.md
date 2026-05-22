# Real-Time Hand Sign to Text & Speech Recognition 🖐️🔊

An end-to-end interactive computer vision and machine learning application that detects hand gestures via a webcam, converts them into text characters in real-time, builds them into words/sentences, and speaks the words aloud using Text-to-Speech (TTS).

This pipeline utilizes **Google MediaPipe** for hand landmark extraction, **Scikit-Learn (Random Forest)** for sign classification, and **pyttsx3** for offline local speech synthesis.

---

## ✨ Features

* **Real-Time Hand Tracking**: Extracts 21 3D hand landmarks at up to 30+ FPS using MediaPipe.
* **Custom Dataset Collector**: Quick script to record your own hand sign images via your webcam.
* **Random Forest Classification**: Robust and fast machine learning classifier with high accuracy.
* **Frame-Stability Detection**: Word building only registers a letter once the gesture is held stable for a set duration, avoiding accidental duplicate letters.
* **Visual HUD**: Includes an elegant overlay displaying the current built sentence and a stability progress bar.
* **Background Threaded TTS**: Seamlessly translates text into speech without freezing or blocking the webcam loop.
* **Keyboard Commands**: Full control over editing, clearing, and speaking your accumulated word/sentence directly from the camera window.

---

## 📂 Project Structure

```text
hand_sign_recognition/
│
├── .gitignore              # Ignores virtual envs, local datasets, and model files
├── requirements.txt        # Project dependencies (OpenCV, MediaPipe, Scikit-learn, pyttsx3)
│
├── collect_data.py         # Step 1: Collect hand gesture images via webcam
├── create_dataset.py       # Step 2: Extract hand landmark coordinates using MediaPipe
├── train_model.py          # Step 3: Train Random Forest classifier and save model
└── recognize_live.py       # Step 4: Real-time hand sign detection, word building & speech
```

---

## 🚀 Step-by-Step Setup & Running

### 1. Clone & Environment Setup
Clone the repository and set up a Python virtual environment:
```bash
# Activate your virtual environment (Windows)
.venv\Scripts\Activate.ps1
# Or (CMD)
.venv\Scripts\activate.bat

# Install all dependencies
pip install -r requirements.txt
```

---

### 2. Pipeline Execution

#### 📂 Step 1: Data Collection (`collect_data.py`)
Collects images for your training dataset. By default, it captures 3 classes (`0`, `1`, `2`) with 100 samples each.
```bash
python collect_data.py
```
* **How to use**: Align your hand for the first gesture in the video window and press `Q` to capture 100 images. Repeat for the remaining gestures.

#### 📊 Step 2: Extract Landmarks (`create_dataset.py`)
Processes your captured raw images through MediaPipe to extract coordinates and normalizes them.
```bash
python create_dataset.py
```
* **Result**: Generates a local `data.pickle` containing structured landmarks and labels.

#### 🧠 Step 3: Train the ML Model (`train_model.py`)
Splits your dataset, trains a **Random Forest Classifier**, and logs the accuracy.
```bash
python train_model.py
```
* **Result**: Generates a trained model file `model.p`.

#### 🎥 Step 4: Run Live Recognition & Speech (`recognize_live.py`)
Launches the webcam feed to detect, build words, and speak them!
```bash
python recognize_live.py
```

---

## 🎮 Live Recognition Controls

Use the following keyboard shortcuts while the camera window is selected:

| Key Command | Action |
| :--- | :--- |
| **Hold Hand Gesture** | Builds the sign (e.g. `0` $\rightarrow$ **A**, `1` $\rightarrow$ **B**, `2` $\rightarrow$ **L**) when held stable for 15 frames. |
| **`SPACEBAR`** | Adds a space between words. |
| **`BACKSPACE`** | Deletes the last character in the built sentence. |
| **`ENTER`** | 🔊 Speaks the current sentence aloud through your speakers. |
| **`C`** | Clears the entire current text. |
| **`Q`** | Exits the live recognition app. |

---

## 🛠️ Built With

* [OpenCV](https://opencv.org/) - Computer Vision & Video Processing
* [MediaPipe](https://github.com/google/mediapipe) - Hand Landmark Tracking
* [Scikit-Learn](https://scikit-learn.org/) - Random Forest Machine Learning Model
* [pyttsx3](https://github.com/nateshmbhat/pyttsx3) - Text-to-Speech Engine
