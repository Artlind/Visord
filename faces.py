import numpy as np
import cv2
import time
import sys
import torch
from model import vgg16
from utils import detect_faces



cap = cv2.VideoCapture(0)

PATH = "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"

# face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(PATH)

path_model = "model_77%acc"
model = vgg16(2, False)
model.load_state_dict(torch.load(path_model, map_location=torch.device("cpu")))
model.eval()
mode = "cascade"

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "cascade":
        faces = face_cascade.detectMultiScale(frame , scaleFactor=1.5, minNeighbors = 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y: y + h, x: x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 0), thickness = 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else :
        faces = detect_faces(model, frame, 10, 200, 5, 5)
        for (x, y),size in faces:
            roi_gray = gray[y: y + size, x: x + size]
            cv2.rectangle(frame, (x, y), (x + size, y + size), (200, 0, 0), thickness = 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
