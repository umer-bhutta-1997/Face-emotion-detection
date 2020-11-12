# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:55:54 2020

@author: umer8
"""

import numpy as np 
import pandas as pd



def loadDataFromCSV(path="fer2013/fer2013.csv"):
    df = pd.read_csv(path)
    pixels = df["pixels"].tolist()
    emotions = pd.get_dummies(df["emotion"])#.as_matrix()
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(" ")]
        face = np.asarray(face).reshape((48,48,1))
        faces.append(face)
    faces = np.asarray(faces)
    return faces/255,emotions
data,emotions = loadDataFromCSV(path="fer2013/fer2013/fer2013.csv")
    



# dataset = pd.read_csv('fer2013/fer2013/fer2013.csv')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Dropout,Flatten,Dense

num_features = 64
num_labels=7
model = Sequential()
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same')) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))    

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(data,emotions,epochs = 3)

import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
from imutils import build_montages

haar_cascade = "haarcascade_frontalface_default.xml"
imgs = os.listdir(path_to_images_dir)
model_name = path_to_model
model = load_model(model_name)
cascade = cv2.CascadeClassifier(haar_cascade)
emotions = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
final_imgs = []

for img_name in imgs:
    img = cv2.imread("face images/"+img_name)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(img_gray, 1.3, 5)

    for (x,y,w,h) in faces:
        detected_face = img_gray[int(y):int(y+h),int(x):int(x+w)]
        detected_face = cv2.resize(detected_face,(48,48),interpolation = cv2.INTER_AREA)
        detected_face = np.array(detected_face).reshape((1,48,48,1))/255.0
        label = np.argmax(model.predict(detected_face))
        emotion_=label
        emotion = emotions[label]
        cv2.putText(img, emotion, (int(x+w/2),y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    final_imgs.append(img)

montages = build_montages(final_imgs,(400,400),(3,2))
for montage in montages:
	cv2.imshow("Montage", montage)
	cv2.waitKey(0)

import cv2 
from tensorflow.keras.models import load_model
import numpy as np

cascade_file = "haarcascade_frontalface_default.xml"
model_file = "fermodel.h5"
emotions = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

face_detection = cv2.CascadeClassifier(cascade_file)
emotion_classifier = load_model(model_file, compile=False)
emotion_=0
cap = cv2.VideoCapture(0)
#capture webcam

while(True):
    ret, img = cap.read()
    faces = face_detection.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        detected_face = img[int(y):int(y+h),int(x):int(x+w)]
        detected_face = cv2.cvtColor(detected_face,cv2.COLOR_BGR2GRAY)
        detected_face = cv2.resize(detected_face,(48,48),interpolation = cv2.INTER_AREA)
        detected_face = np.array(detected_face).reshape((1,48,48,1))/255.0
        label = np.argmax(emotion_classifier.predict(detected_face))
        emotion_=label
        emotion = emotions[label]
        cv2.putText(img, emotion, (x+w,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0), 2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("",img)
    print(emotion_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

# Saving model structure to a JSON file 

model_json = model.to_json() # 
with open("network.json", "w") as json_file: 
    json_file.write(model_json) 

# Saving weights of the model to a HDF5 file 
model.save_weights("network.h5") 
from tensorflow.keras.models import model_from_json
# Loading JSON file 
json_file = open("network.json", 'r') 
loaded_model_json = json_file.read() 
json_file.close() 
loaded_model = model_from_json(loaded_model_json) 

# Loading weights 
loaded_model.load_weights("network.h5") 
loss, accuracy = loaded_model.evaluate(data, emotions) 
