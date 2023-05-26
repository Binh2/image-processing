import cv2
import numpy as np

img = cv2.imread('./data/2.jpg', cv2.IMREAD_UNCHANGED)
print( 'Original Dimensions : ', img.shape)
scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = img # Because my img is already small

# resize image
# resized = cv2.resize(img, dim, interpolation = cv2.inter_area)
# print('resized dimensions : ', resized.shape)
# cv2.imshow("resized image", resized)
# cv2.waitkey(0)
# cv2.destroyallwindows()

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
''' Our classifier returns the ROI of the detected face as a tuple, 
It stores the top left coordinate and the bottom right coordiantes'''
faces = face_classifier.detectMultiScale(gray, 1.0485258, 6)
'''When no faces detected, face_classifier returns and empty tuple'''
if faces is ():
    print("No faces found")

'''We iterate through our faces array and draw a rectangle over each face in faces'''
for (x,y,w,h) in faces:
    cv2.rectangle(resized, (x,y), (x+w,y+h), (127,0,255), 2)
    cv2.imshow('Face Detection', resized)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()