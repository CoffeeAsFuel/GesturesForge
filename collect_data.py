import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

file = open("data.csv", "a", newline="")
writer = csv.writer(file)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in handLms.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

    cv2.imshow("Collect Data", img)

    key = cv2.waitKey(1)

    if key == ord('0'):
        writer.writerow(landmarks + [0])
        print("Saved FIST")

    if key == ord('1'):
        writer.writerow(landmarks + [1])
        print("Saved ONE")

    if key == ord('2'):
        writer.writerow(landmarks + [2])
        print("Saved TWO")

    if key == ord('3'):
        writer.writerow(landmarks + [3])
        print("Saved OPEN")

    if key == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()
