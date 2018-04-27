#!/usr/bin/python2

# Shows live video from webcam with eyes and faces annotated with boxes.

# This file, harpy.py, is just an extremely simple test of face
# detection using video from a webcam. Run it and look into your
# webcam. You should see a green box around your eyes and a blue box
# around your face.

# This is based on haar.py which is a demo file that comes with OpenCV.

import cv2 as cv2
from cv2.cv import *            # For CV_CAP_PROP_...

# CONSTANTS
blue  = (255,0,0)
green = (0,255,0)
red   = (0,0,255)
title = "Haarpy! Hit 'q' to quit."

# Look for the haarcascades examples directory in the current directory.
path = 'haarcascades'
face_cascade=cv2.CascadeClassifier(path+'/haarcascade_frontalface_default.xml')
eye_cascade =cv2.CascadeClassifier(path+'/haarcascade_mcs_eyepair_big.xml')

if (face_cascade.empty() or eye_cascade.empty()):
    print("Oops, couldn't find " + path + "/haarcascade_{frontalface_default,eye}.xml")
    print("Please install the opencv-data package.")
    print("\tapt install opencv-data")
    print("\tcp -a /usr/share/opencv/haarcascades .")
    exit(1)

# Open the camera
capture = cv2.VideoCapture(0)
if (not capture.isOpened()):
    print("Huh. Video device #0 didn't open properly.")
    exit(1)
capture.set(CV_CAP_PROP_FRAME_WIDTH,  100000) # Highest resolution
capture.set(CV_CAP_PROP_FRAME_HEIGHT, 100000)


while True:
    (rv, img) = capture.read()
    if (not rv): break

    # Find a face (can return multiple faces)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # Draw a blue box around the faces
        cv2.rectangle(img,(x,y),(x+w,y+h),blue,3)

        # Find the pair of eyes (can return multiple pairs)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),green,5)

    cv2.imshow(title, cv2.flip(img,1)) # Show it mirrored
    c = chr(cv2.waitKey(25) & 0xFF)
    if (c != '\xff'):   print ("got key: " + c)
    if (c == 'q' or c == '\x1b'):
        break
    if (c == 'p'):
        cv2.imwrite("screenshot.jpg", img)
        print("Wrote image to screenshot.jpg")

capture.release()
cv2.destroyAllWindows()

