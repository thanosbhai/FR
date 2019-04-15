# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 04:41:11 2019

@author: Akash
"""

import cv2

video = cv2.VideoCapture(0)

first_frame = None 

a=1

while(video.isOpened()):

    a=a+1

    check, frame = video.read()
    
    frame = cv2.resize(frame , (800,600))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray,(21,21),0)
        
    if first_frame is None:
        first_frame = gray
        continue
    
    delta_frame = cv2.absdiff(first_frame, gray)
    
    thres_delta= cv2.threshold(delta_frame, 30,4000, cv2.THRESH_BINARY)[1]
    
    thres_delta= cv2.dilate(thres_delta, None, iterations=0)
    
    (_,cnts,_) = cv2.findContours(thres_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y),(x+w,y+h), (20,33,44), 3)
    
    cv2.imshow("frame",frame)
    #cv2.imshow("video",gray)
   # cv2.imshow("delta",delta_frame)
   # cv2.imshow("thres",thres_delta)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

print(a)

video.release()

cv2.destroyAllWindows()

