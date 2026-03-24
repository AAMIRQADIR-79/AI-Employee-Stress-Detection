import cv2
import cvlib as cv
import joblib
import numpy as np
import sqlite3
import time
from datetime import datetime
from collections import deque, Counter
from scipy.io.wavfile import write
import sounddevice as sd
import librosa
import getpass

# -------------------------------
# Step 1: Employee identification
# -------------------------------
employee_name = input("Enter Employee Name: ")

# -------------------------------
# Step 2: Load Emotion Detection Model
# -------------------------------
model_filename = 'emotion_detection_model.pkl'
loaded_model = joblib.load(model_filename)

# -------------------------------
# Step 3: Emotion → Stress Mapping
# -------------------------------
def map_emotion_to_stress(emotion):
    mapping = {
        'happy': 'Low Stress',
        'neutral': 'Moderate Stress',
        'sad': 'High Stress',
        'angry': 'Very High Stress'
    }
    return mapping.get(emotion, 'Unknown')

# -------------------------------
# Step 4: Database Setup
# -------------------------------
conn = sqlite3.connect('employee_emotions.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS emotion_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        face_emotion TEXT,
        voice_tone TEXT,
        typing_speed REAL,
        overall_stress TEXT,
        date TEXT,
        time TEXT
    )
''')
conn.commit()

# -------------------------------
# Step 5: Facial Emotion Detection
# -------------------------------
cap = cv2.VideoCapture(0)
recent_predictions = deque(maxlen=10)

print("\n[INFO] Starting face emotion detection... Press 'q' to stop.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    faces, confidences = cv.detect_face(frame)

    for face, confidence in zip(faces, confidences):
        (start_x, start_y, end_x, end_y) = face
        face_crop = frame[start_y:end_y, start_x:end_x]

        if face_crop.size == 0:
            continue

        face_resize = cv2.resize(face_crop, (100, 100))
        face_flat = face_resize.flatten()

        predicted_emotion = loaded_model.predict([face_flat])[0]
        recent_predictions.append(predicted_emotion)

        stable_emotion = Counter(recent_predictions).most_common(1)[0][0]
        stress_level_face = map_emotion_to_stress(stable_emotion)

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {stable_emotion}", (start_x, start_y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Stress: {stress_level_face}", (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Employee Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[INFO] Stopping emotion detection...\n")
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------
# Step 6: Voice Tone Detection
# -------------------------------
print("\n🎤 Voice Tone Analysis")
print("Please speak for 5 seconds...")

fs = 44100  # Sample rate
seconds = 5
voice = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()
write('employee_voice.wav', fs, voice)

y, sr = librosa.load('employee_voice.wav')
pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
energy = np.mean(librosa.feature.rms(y=y))

if pitch > 200 or energy > 0.05:
    voice_tone = "Stressed"
else:
    voice_tone = "Calm"

print(f"Detected Voice Tone: {voice_tone}")

# -------------------------------
# Step 7: Typing Speed Analysis
# -------------------------------
print("\n⌨️ Typing Speed Test")
sentence = "The quick brown fox jumps over the lazy dog"
print(f"\nType this sentence exactly:\n{sentence}")
input("Press Enter to start typing...")

start_time = time.time()
typed = input("\nNow type here: ")
end_time = time.time()

if typed.strip() == sentence.strip():
    time_taken = end_time - start_time
    words = len(sentence.split())
    wpm = (words / time_taken) * 60
else:
    wpm = 0

print(f"Typing Speed: {wpm:.2f} WPM")

if wpm < 25:
    typing_stress = "High Stress"
elif wpm < 45:
    typing_stress = "Moderate Stress"
else:
    typing_stress = "Low Stress"

print(f"Typing-Based Stress Level: {typing_stress}")

# -------------------------------
# Step 8: Combine All Stress Indicators
# -------------------------------
def combine_stress(face_stress, voice_tone, typing_stress):
    stress_factors = [face_stress, voice_tone, typing_stress]
    high_count = sum(1 for s in stress_factors if "High" in s or "Very High" in s or s == "Stressed")

    if high_count >= 2:
        return "Overall: High Stress"
    elif high_count == 1:
        return "Overall: Moderate Stress"
    else:
        return "Overall: Low Stress"

overall_stress = combine_stress(stress_level_face, voice_tone, typing_stress)

print(f"\n🧠 Final Combined Stress Level: {overall_stress}")

# -------------------------------
# Step 9: Save to Database
# -------------------------------
now = datetime.now()
cursor.execute('''
    INSERT INTO emotion_data (name, face_emotion, voice_tone, typing_speed, overall_stress, date, time)
    VALUES (?, ?, ?, ?, ?, ?, ?)
''', (employee_name, stable_emotion, voice_tone, wpm, overall_stress,
      now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))
conn.commit()
conn.close()

print("\n✅ Data saved successfully to database (employee_emotions.db)")

# -------------------------------
# Step 10: Manager Access (Restricted)
# -------------------------------
print("\n🔒 Manager Access Required to View Data")
manager_password = getpass.getpass("Enter Manager Password: ")

if manager_password == "admin123":  # Change this password as needed
    conn = sqlite3.connect('employee_emotions.db')
    cursor = conn.cursor()
    print("\n--- Employee Emotion and Stress Records ---\n")
    for row in cursor.execute("SELECT * FROM emotion_data"):
        print(row)
    conn.close()
else:
    print("\nAccess Denied ❌ Only authorized manager can view records.")
input("\nPress Enter to exit...")

