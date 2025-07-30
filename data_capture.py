import cv2
import mediapipe as mp
import csv
import os

# Setup hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Make sure 'data' folder exists
os.makedirs("data", exist_ok=True)

# Ask user which sign they're recording
label = input("👉 Enter sign label (e.g. Hello, A, Yes): ")

# Start webcam
cap = cv2.VideoCapture(0)

with open(f"data/{label}.csv", 'a', newline='') as f:
    writer = csv.writer(f)

    while True:
        _, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
                row = []
                for lm in hand.landmark:
                    row += [lm.x, lm.y, lm.z]
                row.append(label)
                writer.writerow(row)

        # Show the video feed
        cv2.imshow("Collecting Data - Press Q to stop", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
