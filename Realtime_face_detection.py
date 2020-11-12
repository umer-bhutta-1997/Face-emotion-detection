#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:59:00 2020

@author: umer
"""

import cv2
import face_recognition
import numpy as np
from keras.preprocessing import image
from keras.models import model_for_json

webcam_videos_stream = cv2.VideoCapture(0)


while True:
    ret, current_frame = webcam_videos_stream.read()
    current_frame_small = cv2.resize(current_frame, (0,0), fx = 0.25, fy = 0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model ="hog")
    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, left_pos, bottom_pos = current_face_location
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        print('Found face {} at top:{}, right:{}, bottom:{}, Left:{}'.format(index+1, top_pos, right_pos, bottom_pos, left_pos))
        # current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]
        # current_face_image = cv2.GaussianBlur(current_face_image, (99,99), 30)
        #burred faec
        # current_frame[top_pos:bottom_pos, left_pos:right_pos] = current_face_image
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        
    cv2.imshow("WebCam Videos", current_frame)
        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam_videos_stream.release()
cv2.destroyAllWindows()
