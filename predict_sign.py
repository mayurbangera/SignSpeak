import cv2
import mediapipe as mp
import numpy as np
import pickle
from gtts import gTTS
from playsound import playsound
import os
import random
import threading
import time

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Speak in background (threaded)
def speak_in_background(text):
    voice_file = f"temp_{random.randint(1, 9999)}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(voice_file)
    playsound(voice_file)
    os.remove(voice_file)

# Text-to-speech memory and stability setup
spoken = ""
last_spoken_time = 0
predictions_list = []

# Setup hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            prediction = model.predict([data])[0]

            # Stability check (smooth prediction)
            predictions_list.append(prediction)
            if len(predictions_list) > 5:
                predictions_list.pop(0)

            if predictions_list.count(prediction) >= 4 and prediction != spoken:
                spoken = prediction
                print(f"🗣️ Speaking: {prediction}")
                threading.Thread(target=speak_in_background, args=(prediction,), daemon=True).start()

            # Show prediction on screen
            cv2.putText(frame, prediction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 3)

    cv2.imshow("SignSpeak - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
