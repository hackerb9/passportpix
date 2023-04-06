#!/usr/bin/python3
# CONFIGURABLE CONSTANTS
global image_ratio, chin_height_ratio, camera_device, camera_rotation
global downscale, frame_downscale

# Width to Height Ratio of required final image size, e.g., 33mm / 48mm
image_ratio=33.0/48.0 

# Distance from chin to bottom of picture divided by picture length
chin_height_ratio=7.0/48.0

# Which camera to open (first is 0)
camera_device=0

# Is camera on its side? 0 == nope, 1 == 90, 2 == 180, 3 == 270.
camera_rotation=0

# What resolution in pixels to downscale the image to (max width, height)
# This is used both to speed up the Haar cascade and for display on the screen.
# It does not affect the final output image resolution. 
# Set to 0 to not downscale.  A value of 320 is reasonable for slow machines.
downscale=320
#downscale=0
frame_downscale=None
######################################################################

import cv2 as cv2
import numpy as np

# My files
from fps import FPS
from eprint import eprint


# Constants
blue  = (255,0,0)
green = (0,255,0)
red   = (0,0,255)

title = "Hit space to save passport.jpg, q to quit"


def maxpect(image_ratio, old_width, old_height):
    # Rescale camera size to the maximum aspect ratio it'll fit.
    # Input: image_ratio == desired width/height
    #	     old_width, old_height == current width/height
    # Output: (new_width, new_height)
    #
    # If old_width or old_height == 0, then 320 pixels will be used.

    # Testing. Should this bomb out on zero frame size?
    if ( old_width <= 0) or (old_height <= 0): return None

    if ( old_width <= 0):  old_width = 320
    if (old_height <= 0): old_height = 320

    if (image_ratio < old_width/old_height):
        new_height=old_height
        new_width=int(old_height*image_ratio)
    else:
        new_width=old_width
        new_height=int(old_width/image_ratio)

    if (new_width>old_width):
        new_height=new_height*old_width/new_width
        new_width=old_width

    if (new_height>old_height):
        new_width=new_width*old_height/new_height
        new_height=old_height
        
    return (new_width, new_height)

def init():
    # Create a window and make it fullscreen
    cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Initialize the camera and Haar cascades as global variables
    global face_cascade, eye_cascade, capture
    global frame_width, frame_height
    global fps, downscale, frame_downscale

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
    for backend in (cv2.CAP_V4L2, cv2.CAP_V4L, cv2.CAP_GPHOTO2, cv2.CAP_FFMPEG, cv2.CAP_PVAPI, cv2.CAP_GSTREAMER, None):
        capture = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
        if capture: break


    if (not capture.isOpened()):
        print("Oops. Video device %d didn't open properly. Dying."
              % (camera_device))
        exit(1)

    print("Successfully opened video device %d using %s."
          % (camera_device, capture.getBackendName()))
        
    # Instead of returning an error code, thrown an exception on errors. 
    capture.setExceptionMode(True)

    # Set camera to max resolution.
    # 	Oops, this works with some backends but makes others go haywire. 
    # 	Probably best to not muck with the resolution in CV2.
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,10000)
    except:
        capture = cv2.VideoCapture(camera_device)

    frame_width=float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height=float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (frame_width==0 or frame_height==0):
        print("Error. Video device #%d returned resolution of %d x %d. Dying."
              % (camera_device, frame_width, frame_height))
        exit(1)
        
    if (camera_rotation==1 or camera_rotation==3):
        frame_width, frame_height = frame_height, frame_width

    print("Capturing at %d x %d" % (frame_width, frame_height) )

    print("Output image size will be %d x %d" %
          maxpect(image_ratio, frame_width, frame_height))

    # frame_downscale is the (width, height) for Haar processing and display.
    # (Same aspect ratio as the frame_width/frame_height, but fits in
    # a square of length downscale).
    # See also the 'downscale' global variable at the top of this file.
    recalculate_frame_downscale(downscale)

    # Read at least one image before starting the FPS counter 
    capture.read()
    fps=FPS()

    # Give some hints to the user on stdout
    print("Press space to save passport.jpg, q to quit.")


# Static vars for findtheface so image doesn't jump back if we lose the face.
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

def centerandscale(img, x_y_w_h, ex_ey_ew_eh):
    x,y,w,h = x_y_w_h
    ex,ey,ew,eh = ex_ey_ew_eh

    # Given the image and bounding boxes for the face and eyes,
    # recenter the image and scale it so the eyes are in the right place.

    height, width, channels = img.shape

    heightofchin=y+h                      # Bottom of bounding box.
    heightofeyes=((y+ey) + (y+ey+eh))/2.0 # Eyes are relative to face box
    chintoeyes=abs(heightofchin - heightofeyes)

    # The eyes are in the middle and there is a chin_height_ratio
    # percentage gap between chin and bottom of picture. (E.g.,
    # 1/7th). That means, the distance from the chin to the eyes,
    # chintoeyes, should be, once scaled, be equal to
    # (1/2- chin_height_ratio) times the image height. (E.g., 5/14th).

    scale = (0.5-chin_height_ratio) * height/chintoeyes
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
    tx = width/2-(x+ex+(x+ex+ew))/2
    ty = height/2-(y+ey+(y+ey+eh))/2
    M = np.float32([[1,0,tx],[0,1,ty]])
    img = cv2.warpAffine(img,M,(width,height))
    return img

def crop(img):
    # Given an image, and the global variable image_ratio
    # return image cropped to correct aspect ratio, centered on the original.
    height, width, channels = img.shape
    (final_width, final_height) = maxpect(image_ratio, width, height)
    tx=(final_width-width)/2
    ty=(final_height-height)/2
    M = np.float32([[1,0,tx],[0,1,ty]])
    img = cv2.warpAffine(img,M,(final_width,final_height))
    return img

def recalculate_frame_downscale(downscale):
    # Sets frame_downscale based on frame_width, frame_height, and downscale.

    # frame_width/height is the size of the video capture frame.
    # downscale is the maximum width or height that should be used. 

    # frame_downscale is the (width, height) for Haar processing and display.
    # It has the same aspect ratio as the video capture frame
    # (frame_width / frame_height), but fits in a square of length downscale.

    # When downscale is zero, frame_downscale will not be used, so we
    # set it to None so it will (hopefully) cause an error upon bugs.

    global frame_downscale
    global frame_width, frame_height
#    global oldfaces, oldeyes

    if downscale == 0:
        frame_downscale = None
        print("No longer downscaling images")
    else:
        frame_downscale = maxpect(frame_width/frame_height, downscale, downscale)
        print("Downscaled size for internal processing is now %d x %d"
              % frame_downscale)

    # Invalidate found face (necessary?)
#    oldfaces = None
#    oldeyes = None

    return 

    
def main():    
    global downscale, frame_downscale

    init()

    while True:
        (rv, original) = capture.read()
        if (not rv): break

        fps.incrementFrames()

        # Rotate the image, if the camera is on its side
        if (camera_rotation):
            original=np.rot90(original, camera_rotation)

        # Downscale image to make findtheface() faster
        if frame_downscale:
            img = cv2.resize(original, frame_downscale)
            imgscale=float(original.shape[0])/img.shape[0]
        else:
            img = original
            imgscale=1

        # Find the face and eyes using the Haar cascade
        (scaledface, scaledeyes) = findtheface(img)

        # If both are found, center on the eyes and scale
        if (scaledface is not None and scaledeyes is not None):
            if downscale:
                face = np.dot( scaledface[0], imgscale )
                eyes = np.dot( scaledeyes[0], imgscale )
            else:
                face = scaledface[0]
                eyes = scaledeyes[0]

            img=centerandscale(original, face, eyes)

            # Crop to the proper aspect ratio
            img=crop(img)

        # Show the image (and frames per second)
        cv2.imshow(title, cv2.flip(img,1))
        if (fps.framecount % 10 == 0):
            eprint('%.2f fps' % fps.getFPS(), end='\r')
            fps.reset()

        # Show image and wait for a key
        c = chr(cv2.waitKey(1) & 0xFF)

        if (c == 'q' or c == '\x1b'): 		 # q or ESC to quit
            break
        if (c == ' ' or c=='p' or c=='s'):       # Print a screenshot
            img=centerandscale(original, face, eyes)
            img=crop(img)
            cv2.imwrite("passport.jpg", img)
            print("Wrote image to passport.jpg")
        if (c == '1'): 
            if downscale:
                downscale = 0
            else:
                downscale = 320
            recalculate_frame_downscale(downscale)

    capture.release()
    cv2.destroyAllWindows()


main()

