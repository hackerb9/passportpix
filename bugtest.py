#!/usr/bin/python3

# Exercise the fullscreen bug in OpenCV 4.6

import cv2 as cv2
from cv2 import *            # For CAP_PROP_...
import numpy as np

# CONSTANTS
blue  = (255,0,0)
green = (0,255,0)
red   = (0,0,255)
title = "Haarpy! Hit 'q' to quit."

print ("""Test of High-level GUI bugs in OpenCV 4.5 and 4.6. Interaction
between fullscreen and resize is unclear and perhaps buggy.

In 4.6, resizing the window while in fullscreen will turn off
fullscreen. It ought to stay in fullscreen and remember the new window
size for when fullscreen is turned off. To test, press 5 r and notice
that resizing the window to 500x500 works correctly when not
fullscreen. Now press f 6 r and notice that as soon as resizeWindow()
is called, fullscreen was turned off. Now try f 6 shift+R, which
attempts to enable fullscreen mode immediately after resizeWindow()
and note that it fails: the screen stays windowed.

In 4.5, windows cannot be made resized smaller. To test, press 5 r 6 r
and notice that the window can resize larger. Now press 4 r 3 r, the
window fails to get smaller.

""")

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

 # Capture at highest resolution. (Gstreamer backend is buggy).
if (capture.getBackendName() != "GSTREAMER"):
    capture.set(CAP_PROP_FRAME_WIDTH,  100000)
    capture.set(CAP_PROP_FRAME_HEIGHT, 100000)

# Create a window that can be made fullscreen
cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)

# Default size to crop image shown on screen
cropsize=320

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

    # Crop image
    height, width, channels = img.shape
    desired_width, desired_height = (cropsize, cropsize) \
        if cropsize else (width, height)
    tx = ( desired_width-width ) / 2
    ty = ( desired_height-height ) / 2
    M = np.float32( [[1,0,tx], [0,1,ty]] )
    img = cv2.warpAffine( img, M, ( desired_width, desired_height) )

    # Show the image, mirrored
    cv2.imshow(title, cv2.flip(img,1))

    # Check for user keystroke
    k = cv2.waitKey(1)
    if (k == -1):
        continue

    c = chr(k & 0xFF)
    print( c )

    if (c == 'q' or c == '\x1b'):
        print ( '' )
        break

    elif (c == 'p' or c == ' '):
        cv2.imwrite("screenshot.jpg", img)
        print("\tWrote image to screenshot.jpg")

    elif (c == 'f'):          		# Toggle full screen
        isFull = cv2.getWindowProperty( title, cv2.WND_PROP_FULLSCREEN )
        print( f"\tToggling fullscreen from {isFull} to {1-isFull}." )
        cv2.setWindowProperty( title, cv2.WND_PROP_FULLSCREEN, 1 - isFull )

    elif (ord('0') <= ord(c) and ord(c) <= ord('9')  ): # Change cropsize
        cropsize = int(c) * 100
        print (f"\tCropping to {cropsize} x {cropsize}")

    elif (c == 'g'):          		# Get current status
        print( f"\tGet current window info: ", end=None )
        isFull = cv2.getWindowProperty( title, cv2.WND_PROP_FULLSCREEN )
        print( f"WND PROP_FULLSCREEN is {isFull}, ", end=None )
        (dummy, dummy, width, height) = cv2.getWindowImageRect( title )
        print( f"Size is {width, height}" )

    elif (c == 'r'):          		# Resize window to cropsize
        (dummy, dummy, width, height) = cv2.getWindowImageRect(title)
        print( f"\tResizing from {width, height} to {cropsize, cropsize}." )
        cv2.resizeWindow( title, (cropsize, cropsize) )

    elif (c == 'R'):          		# Resize and enable fullscreen
        print( f"\tResizing to {cropsize, cropsize} and enabling fullscreen." )
        cv2.resizeWindow( title, (cropsize, cropsize) )
        cv2.setWindowProperty( title, cv2.WND_PROP_FULLSCREEN, 1 )

# When user hits 'q', main loop exits
capture.release()
cv2.destroyAllWindows()

