#import open cv library

import cv2

from random import randrange as r

#dataset load
trainedData=cv2.CascadeClassifier('Traindata.xml')

#start the webcam
webcam=cv2.VideoCapture(0)

while True:
    success,img=webcam.read()

#choose an image
#img=cv2.imread('picture.jpg')

    #conversion image to grayscale
    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #detect faces
    faceCoordinates = trainedData.detectMultiScale(grayimg)
    print(faceCoordinates)
    for x,y,w,h in faceCoordinates:
        cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)
    
#show this image
    cv2.imshow('Window',img)
    key=cv2.waitKey(1)
    if (key==81 or key==113):
        break


webcam.release()
print('End of program')


