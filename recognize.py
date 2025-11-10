from sklearn.neighbors import KNeighborsClassifier
import cv2, pickle, numpy as np
import time
from datetime import datetime
import csv
import os

from win32com.client import Dispatch

def speak(strl):
    speak=Dispatch("SAPI.SpVoice")
    speak.Speak(strl)

# Ensure attendance folder exists
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# Load data
with open('data/names.pkl', 'rb') as f:
    names = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    faces = pickle.load(f)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces, names)

# Load UI frame
imgBackground = cv2.imread("background.png")

COL_NAMES = ['NAME', 'TIME']

# Start camera
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

while True:
    ok, frame = video.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected:
        crop = frame[y:y+h, x:x+w]
        resized = cv2.resize(crop, (50, 50)).reshape(1, -1)

        pred = knn.predict(resized)[0]   # Name
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        attendance = [pred, timestamp]

        # Drawing on frame
        cv2.putText(frame, pred, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

    # Put frame into background UI
    imgBackground[162:162+480, 55:55+640] = frame
    cv2.imshow("Recognize", imgBackground)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('o'):
        speak("Attendance Taken...")
        time.sleep(5)
        file_path = f"Attendance/Attendance_{date}.csv"
        
        file_exists = os.path.isfile(file_path)

        with open(file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)
        print("✅ Attendance Saved:", attendance)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
