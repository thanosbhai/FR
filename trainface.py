# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:10:02 2019

@author: Akash
"""

import os
from PIL import Image
import numpy as np
import cv2
import pickle

base_dir = os.path.dirname(os.path.abspath("__file__"))

img_dir = os.path.join(base_dir, "C:\\Users\\Akash\\.spyder-py3\\x")

face = cv2.CascadeClassifier("E:\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ",".").lower()
            #print(label,path)
            
            if label in label_ids:
                pass
            else:
                label_ids[label]= current_id
                current_id += 1
                
            id = label_ids[label]
            #print(label_ids)
            
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            
            image_array = np.array(pil_image, "uint8")
            #print(image_array)
            
            faces = face.detectMultiScale(image_array, scaleFactor=1.05, minNeighbors=5)
            
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id)
                
#print(y_labels)
#print(x_train)
               
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)
    
recognizer.train(x_train ,np.array(y_labels))
recognizer.save("trainner.yml")
    