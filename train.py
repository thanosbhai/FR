# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:00:22 2019

@author: Akash
"""

import cv2
import pickle

face = cv2.CascadeClassifier("E:\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
eye = cv2.CascadeClassifier("E:\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_eye.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels= {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

video = cv2.VideoCapture(0)


while(video.isOpened()):


    check, frame = video.read()
    
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    
    for x,y,w,h in faces:
        
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        id, conf = recognizer.predict(roi_gray)
        
        if conf>=45 and conf <=85:
            print(id)
            print(labels[id])
            cv2.putText(frame, labels[id], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (129,198,92), 3, cv2.LINE_AA)
            
        img_item = "C:\\Users\\Akash\\.spyder-py3\\myimage.png"
        cv2.imwrite(img_item, roi_color)
        
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h), (130,23,43), 3)
        
        eyes= eye.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
        
        #for ex,ey,ew,eh in eyes:
        #    frame = cv2.rectangle(frame, (ex,ey),(ex+ew,ey+eh), (120,23,43), 3)
        
    
    cv2.imshow("video",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()

cv2.destroyAllWindows()

