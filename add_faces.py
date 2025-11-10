import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datatime

os.makedirs('data', exist_ok=True)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0
name = input("Enter your Name: ").strip()

while True:
    ok, frame = video.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        try:
            resized = cv2.resize(crop_img, (50, 50))
        except:
            continue
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized)
        i += 1

        cv2.putText(frame, f"{len(faces_data)}/100", (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("Add Faces", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)          # shape: (100, 50, 50, 3)
faces_data = faces_data.reshape(100, -1)     # shape: (100, 7500)

# ----- Save names -----
names_path = 'data/names.pkl'
if not os.path.exists(names_path):
    names = [name] * 100
else:
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
with open(names_path, 'wb') as f:
    pickle.dump(names, f)

# ----- Save faces -----
faces_path = 'data/faces_data.pkl'
if not os.path.exists(faces_path):
    faces = faces_data
else:
    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)               # shape: (N, 7500)
    faces = np.append(faces, faces_data, axis=0)
with open(faces_path, 'wb') as f:
    pickle.dump(faces, f)

print("Saved:", faces.shape, "samples; names:", len(names))
