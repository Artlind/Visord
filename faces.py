import numpy as np
import cv2
import time
import sys

cap = cv2.VideoCapture(0)

PATH = "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"

# face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(PATH)


while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame , scaleFactor=1.5, minNeighbors = 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y: y + h, x: x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 0), thickness = 3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
