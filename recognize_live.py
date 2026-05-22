import pickle
import time
import threading

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# ── Load trained model ──────────────────────────────────────────────
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# ── Text-to-Speech (runs in a separate thread to avoid blocking) ──
def speak(text):
    """Speak text in a background thread with a fresh engine instance."""
    def _speak():
        try:
            tts = pyttsx3.init()
            tts.setProperty('rate', 150)
            tts.setProperty('volume', 1.0)
            tts.say(text)
            tts.runAndWait()
            tts.stop()
        except Exception as e:
            print(f"  ⚠ TTS error: {e}")
    t = threading.Thread(target=_speak, daemon=True)
    t.start()

# ── Camera & MediaPipe setup ───────────────────────────────────────
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# ── State for word building ────────────────────────────────────────
sentence = ""                   # accumulated text
last_character = None           # character detected in previous frame
stable_count = 0                # how many consecutive frames same char was seen
STABLE_THRESHOLD = 15           # frames needed before a letter is "locked in"
letter_added = False            # prevents adding the same held sign repeatedly
last_add_time = 0               # timestamp of last letter addition
ADD_COOLDOWN = 1.0              # seconds to wait before same letter can repeat

print("=" * 55)
print("  HAND SIGN → WORD → SPEECH")
print("=" * 55)
print("  Hold a sign steady to add a letter.")
print("  SPACE  = add a space between words")
print("  BACKSPACE = delete last character")
print("  ENTER  = speak the sentence aloud")
print("  C      = clear all text")
print("  Q      = quit")
print("=" * 55)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    predicted_character = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        hand_landmarks = results.multi_hand_landmarks[0]
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # ── Draw bounding box and predicted letter ──
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # ── Stability logic: lock in a letter after holding it steady ──
    if predicted_character is not None:
        if predicted_character == last_character:
            stable_count += 1
        else:
            stable_count = 1
            letter_added = False
        last_character = predicted_character

        if stable_count >= STABLE_THRESHOLD and not letter_added:
            now = time.time()
            if now - last_add_time >= ADD_COOLDOWN:
                sentence += predicted_character
                letter_added = True
                last_add_time = now
                print(f"  ✓ Added '{predicted_character}'  →  \"{sentence}\"")
    else:
        # No hand detected → reset stability so next sign starts fresh
        stable_count = 0
        last_character = None
        letter_added = False

    # ── Draw HUD overlay ────────────────────────────────────────────
    # Dark banner at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 90), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Current sentence
    display_text = sentence if sentence else "(show a sign to start)"
    cv2.putText(frame, display_text, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2, cv2.LINE_AA)

    # Stability progress bar
    progress = min(stable_count / STABLE_THRESHOLD, 1.0)
    bar_w = int(progress * (W - 30))
    bar_color = (0, 255, 0) if progress >= 1.0 else (0, 180, 255)
    cv2.rectangle(frame, (15, 60), (15 + bar_w, 75), bar_color, -1)
    cv2.rectangle(frame, (15, 60), (W - 15, 75), (100, 100, 100), 1)

    # Controls hint at bottom
    cv2.putText(frame, "SPACE:space  BKSP:delete  ENTER:speak  C:clear  Q:quit",
                (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
                cv2.LINE_AA)

    cv2.imshow('Hand Sign Recognition', frame)

    # ── Keyboard controls ───────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        sentence += " "
        print(f"  ✓ Added SPACE  →  \"{sentence}\"")
    elif key == 8:  # Backspace
        sentence = sentence[:-1]
        print(f"  ✗ Deleted  →  \"{sentence}\"")
    elif key in (13, 10):  # Enter → speak
        if sentence.strip():
            print(f"  🔊 Speaking: \"{sentence.strip()}\"")
            speak(sentence.strip())
    elif key == ord('c'):
        sentence = ""
        print("  ✗ Cleared all text.")

cap.release()
cv2.destroyAllWindows()
