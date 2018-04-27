#!/usr/bin/python2
import cv2 as cv2
import numpy as np

# CONSTANTS

# Width to Height Ratio of required final image size, e.g., 33mm / 48mm
image_ratio=33.0/48.0 

# Distance from chin to bottom of picture divided by picture length
chin_height_ratio=7.0/48.0


blue  = (255,0,0)
green = (0,255,0)
red   = (0,0,255)


def maxpect(image_ratio, frame_width, frame_height):
    # Rescale camera size to the maximum aspect ratio it'll fit.

    if (image_ratio < frame_width/frame_height):
        final_height=frame_height
        final_width=int(frame_height*image_ratio)
    else:
        final_width=frame_width
        final_height=int(frame_width/image_ratio)

    if (final_width>frame_width):
        final_height=final_height*frame_width/final_width
        final_width=frame_width

    if (final_height>frame_height):
        final_width=final_width*frame_height/final_height
        final_height=frame_height
        
    return (final_width, final_height)

def init():
    # Initialize the camera and Haar cascades
    global face_cascade, eye_cascade, capture
    global frame_width, frame_height, final_width, final_height

    # Load up the sample Haar cascades from opencv-data (or current directory)
    for path in ('.', 'haarcascades', '/usr/local/share/opencv/haarcascades/',
                 '/usr/share/opencv/haarcascades/'):
        
        face_cascade = cv2.CascadeClassifier(
                           path + '/haarcascade_frontalface_default.xml' )
        eye_cascade  = cv2.CascadeClassifier(
                           path + '/haarcascade_mcs_eyepair_big.xml' )
        if (not face_cascade.empty() and not eye_cascade.empty()): break

    if (face_cascade.empty() or eye_cascade.empty()):
        print("Oops, couldn't find haarcascade_frontalface_default.xml")
        print("or haarcascade_mcs_eyepair_big.xml")
        print("Please install the opencv-data package.")
        print("\tapt install opencv-data")
        print("\tcp -a /usr/share/opencv/haarcascades .")
        exit(1)

    # Open the camera
    capture = cv2.VideoCapture(0)
    if (not capture.isOpened()):
        print("Huh. Video device #0 didn't open properly. Dying.")
        exit(1)

    # Set camera to max resolution -- too slow and unneeded?
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 100000)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,100000)

    frame_width=int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height=int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    print("Capturing at %d x %d\n" % (frame_width, frame_height) )

    (final_width, final_height)=maxpect(image_ratio, frame_width, frame_height)


    print("Output image size will be %d x %d\n" % (final_width, final_height) )


oldfaces=None
oldeyes=None
def findtheface(img):
    # Given the BGR image 'img', draw boxes on it for eyes and face,
    # then return the coordinate for (face, eyes).
    # Or (None, None) if there is not exactly one face and one pair of eyes.

    global oldfaces, oldeyes

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # Draw a blue box around it, if we found exactly one face.
        if (len(faces) == 1):
            cv2.rectangle(img,(x,y),(x+w,y+h),blue,3)
        else:
            cv2.line(img,(x,y),(x+w,y+h),red,1)
            cv2.line(img,(x+w,y),(x,y+h),red,1)

        # Find a pair of eyes (can return multiple pairs)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            # Draw a green box, if we found exactly one pair of eyes.
            if (len(eyes) == 1):
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),green,5)
            else:
                cv2.line(roi_color,(ex,ey),(ex+ew,ey+eh),red,1)
                cv2.line(roi_color,(ex+ew,ey),(ex,ey+eh),red,1)

    if (len(faces)==1 and len(eyes)==1):
        oldfaces=faces
        oldeyes=eyes
        return (faces, eyes)
    else:
        if (oldfaces is None or oldeyes is None):
    	    return (None, None)
        else:
            return (oldfaces, oldeyes)

def centerandscale(img, (x, y, w, h), (ex,ey,ew,eh)):
    # Given the image and bounding boxes for the face and eyes,
    # recenter the image and scale it so the eyes are in the right place.

    heightofchin=y+h                      # Bottom of bounding box.
    heightofeyes=((y+ey) + (y+ey+eh))/2.0 # Eyes are relative to face box
    chintoeyes=abs(heightofchin - heightofeyes)

    # The eyes are in the middle and there is a chin_height_ratio
    # percentage gap between chin and bottom of picture. (E.g.,
    # 1/7th). That means, the distance from the chin to the eyes,
    # chintoeyes, should be, once scaled, be equal to
    # (1/2- chin_height_ratio) times the frame height. (E.g., 5/14th).

    scale = (0.5-chin_height_ratio) * frame_height/chintoeyes
    img = cv2.resize(img, None, fx=scale, fy=scale)

    # This is silly. How do I numptify this?
    x=scale*x
    y=scale*y
    w=scale*w
    h=scale*h
    ex=scale*ex
    ey=scale*ey
    ew=scale*ew
    eh=scale*eh

    # Translation needed in the X and Y directions to put eyes in center
    tx = frame_width/2-(x+ex+(x+ex+ew))/2
    ty = frame_height/2-(y+ey+(y+ey+eh))/2
    M = np.float32([[1,0,tx],[0,1,ty]])
    img = cv2.warpAffine(img,M,(frame_width,frame_height))
    return img

def crop(img):
    # Given an image, and the global variables final_width & _height,
    # return image cropped to final resolution, centered on the original.
    tx=(final_width-frame_width)/2
    ty=(final_height-frame_height)/2
    M = np.float32([[1,0,tx],[0,1,ty]])
    img = cv2.warpAffine(img,M,(final_width,final_height))
    return img

def main():    
    init()

    while True:
        (rv, img) = capture.read()
        if (not rv): break

        original=img.copy()
        (face, eyes) = findtheface(img)

        if (face is not None and eyes is not None):
            img=centerandscale(img, face[0], eyes[0])

        img=crop(img)

        cv2.imshow("Hit space to save passport.jpg, q to quit", cv2.flip(img,1))
        c = chr(cv2.waitKey(25) & 0xFF)
        if (c == 'q' or c == '\x1b'):
            break
        if (c == ' ' or c=='p' or c=='s'):       # Print a screenshot
            img=centerandscale(original, face[0], eyes[0])
            img=crop(img)
            cv2.imwrite("passport.jpg", img)
            print("Wrote image to passport.jpg")

    capture.release()
    cv2.destroyAllWindows()


main()

