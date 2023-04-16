#!/usr/bin/python3
# CONFIGURABLE CONSTANTS - .-. -.--
global photo_width, photo_height, photo_aspect
global eye_distance, eye_height
global chin_height, use_chin_scaling
global camera_device, camera_rotation, camera_codec, camera_fps
global downscale

# Photo metadata (Note: not saved in .jpg file, yet)
photo_width=2.0
photo_height=2.0
photo_units="in"

# Width to Height Ratio of required final photo size, e.g., 33mm / 48mm
# US Passport is 2in / 2in
photo_aspect=photo_width/photo_height

# Distance between eyes as a fraction of the picture width (US Passport)
eye_distance = 2.0 / 12.0

# Distance from eyes to the bottom of the picture divided by picture length
eye_height = 7.0 / 12.0					#  (US Passport)

# Distance from eyes to the bottom of the picture divided by picture length
#eye_height = 24.0 / 48.0				#  (CN Visa)

# Distance from chin to bottom of picture divided by picture length (CN Visa)
chin_height=7.0/48.0
use_chin_scaling=False           # False (default) means to use eye_distance

# Which camera to open (first is 0)
camera_device=0

# Is camera on its side? 0 == nope, 1 == 90, 2 == 180, 3 == 270.
camera_rotation=0

# Show the image like in a mirror (flipped horizontal)
camera_mirrored=1

# Max resolution (in pixels) to downscale the image to before processing. 
# This is used both to speed up the Haar cascade and for display on the screen.
# It does not affect the final output image resolution. 
# Set to 0 to not downscale.  A value of 640 is reasonable for slow machines.
downscale=640
frame_downscale=None

# Frames per second to request the camera to retrieve. None means maximum.
# While unlikely, it is possible you'd want to set this lower on a bandwidth
# limited camera as there's a small chance it'd improve compression quality.
camera_fps=None

# Optionally use a different codec for getting data from the camera.
# For example, codec='MJPG' might improve FPS and resolution over USB2,
# but the poor quality from compression makes it a questionable tradeoff.
camera_codec=None

######################################################################

import cv2 as cv2
import numpy as np
from math import sqrt
import sys

# My files
from fps import FPS
from eprint import eprint


# Constants
blue  = (255,0,0)
green = (0,128,0)
red   = (0,0,255)
yellow   = (0,255,255)
magenta   = (255,0,255)

title = "Hit space to save passport.jpg, q to quit"


def maxpect(photo_aspect, old_width, old_height):
    # Calculate the maximum frame size that maintains the aspect ratio.
    #
    # Input: photo_aspect == desired width/height
    #	     old_width, old_height == current width/height
    # Output: (new_width, new_height)


    # Bomb out on zero frame size.
    if ( old_width <= 0) or (old_height <= 0): return None

    if (photo_aspect < old_width/old_height):
        new_height=old_height
        new_width=int(old_height*photo_aspect)
    else:
        new_width=old_width
        new_height=int(old_width/photo_aspect)

    if (new_width>old_width):
        new_height=new_height*old_width/new_width
        new_width=old_width

    if (new_height>old_height):
        new_width=new_width*old_height/new_height
        new_height=old_height
        
    return (int(new_width), int(new_height))

def init():
    # Initialize camera, UI window, Haar cascades, etc.

    # Initialize the camera and Haar cascades as global variables
    global face_cascade, eyepair_cascade, lefteye_cascade, righteye_cascade
    global capture, frame_width, frame_height
    global fps, downscale, frame_downscale
    global face_warp

    face_warp = None     # Transform matrix to remap face in image

    # Load up the sample Haar cascades from opencv-data (or current directory)
    for path in ('.', 'haarcascades', '/usr/local/share/opencv/haarcascades/',
                 '/usr/share/opencv/haarcascades/'):
        
        face_cascade = cv2.CascadeClassifier(
                           path + '/haarcascade_frontalface_default.xml' )
        eyepair_cascade  = cv2.CascadeClassifier(
                           path + '/haarcascade_mcs_eyepair_big.xml' )
        # NB: The OpenCV haar cascades use "LEFT" to mean "Left relative
        # to the person in the image", while we mean left-side of image .
        lefteye_cascade  = cv2.CascadeClassifier(
                           path + '/haarcascade_righteye_2splits.xml' )
        righteye_cascade  = cv2.CascadeClassifier(
                           path + '/haarcascade_lefteye_2splits.xml' )

        if not (face_cascade.empty() or eyepair_cascade.empty() or
                lefteye_cascade.empty() or righteye_cascade.empty()):
            break

    # Did we find all the 
    if (face_cascade.empty() or eyepair_cascade.empty()):
        print("Oops, couldn't find haarcascade_frontalface_default.xml")
        print("or haarcascade_mcs_eyepair_big.xml")
        print("Please install the opencv-data package.")
        print("\tapt install opencv-data")
        print("\tcp -a /usr/share/opencv/haarcascades .")
        exit(1)

    # Open the camera, but don't use GSTREAMER.
    for backend in (cv2.CAP_DSHOW, cv2.CAP_GPHOTO2, cv2.CAP_V4L2, cv2.CAP_V4L, cv2.CAP_FFMPEG, cv2.CAP_PVAPI, cv2.CAP_GSTREAMER, None):
        capture = cv2.VideoCapture(camera_device, backend)
        if capture and capture.isOpened(): break

    if (not capture.isOpened()):
        eprint(f"Error. Could not open video device {camera_device}.")
        exit(1)

    print("Successfully opened video device %d using %s."
          % (camera_device, capture.getBackendName()))
        
    if camera_codec:
        if set_fourcc(capture, camera_codec):
            print( f"Successfully changed camera codec to '{camera_codec}'." )
        else:
            print(f"Warning: Couldn't change camera codec to '{camera_codec}'.")
    print( "Camera codec is ", get_fourcc(capture) )

    if camera_fps:
        capture.set(cv2.CAP_PROP_FPS, camera_fps)
        if camera_fps == capture.get(cv2.CAP_PROP_FPS):
            print(f"Camera FPS successfully set to {camera_fps}.")
        else:
            print(f"Warning: Could not change camera FPS to {camera_fps}.")

    # Instead of returning an error code, throw an exception on errors. 
    capture.setExceptionMode(True)

    # Set camera to max resolution.
    #
    # 	NOTA BENE: this works with some backends, such as V4L2,
    #   but makes others, such as GSTREAMER, go haywire. 
    #
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,10000)
    except:
        # Gstreamer gets so confused the camera has to be reopened.
        capture = cv2.VideoCapture(camera_device, backend)

    frame_width=float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height=float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (frame_width==0 or frame_height==0):
        eprint("Error. Video device #%d returned resolution of %d x %d. Dying."
              % (camera_device, frame_width, frame_height))
        exit(1)
        
    if (camera_rotation==1 or camera_rotation==3):
        frame_width, frame_height = frame_height, frame_width

    print("Capturing at %d x %d at %d FPS." %
          (frame_width, frame_height, capture.get(cv2.CAP_PROP_FPS) ))

    print("Output image size will be %d x %d" %
          maxpect(photo_aspect, frame_width, frame_height))

    # frame_downscale is the (width, height) for Haar processing and display.
    # (Same aspect ratio as the frame_width/frame_height, but fits in
    # a square of length downscale).
    recalculate_frame_downscale(downscale)

    # Read at least one image before starting the FPS counter 
    capture.read()
    fps=FPS()

    # Create a window. User can hit 'f' to make it fullscreen.
    cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
    cv2.resizeWindow(title, frame_downscale)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, 0)

    # Give some hints to the user on stdout
    print("Press space to save passport.jpg, q to quit.")


# Static vars to cache findtheface so image doesn't jump back if we lose the face.
oldface=None
oldeyes=None
def findtheface(img):
    # Given the BGR image 'img', draw boxes on it for eyes and face,
    # then return the coordinate for (face, eyes).
    # Or (None, None) if there is not exactly one face and one pair of eyes.

    global oldface, oldeyes

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = None            # Array of rectangles for detected pair of eyes

    for (x,y,w,h) in faces:
        if (len(faces) != 1):	 	# Wrong number of faces
            cv2.line(img,(x,y),(x+w,y+h),blue,1)
            cv2.line(img,(x+w,y),(x,y+h),blue,1)
            eyes = None
            continue

        # Draw a blue box around it, if we found exactly one face.
        cv2.rectangle(img,(x,y),(x+w,y+h),blue,3)
        
        # Find a pair of eyes in this face (can return multiple pairs!)
        # (Fairly robust)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eyepair_cascade.detectMultiScale(roi_gray)
        if (len(eyes) != 1):
            for e in eyes:
                exOut(roi_color,e,green,1)
            return (oldface, oldeyes)

        # Draw a green box, if we found exactly one pair of eyes.
        cv2.rectangle(roi_color,eyes[0],green,2)
        (ex,ey,ew,eh) = eyes[0]

    if (len(faces)==1 and len(eyes)==1):
        (oldface, oldeyes) = (faces[0], eyes[0])
        return (faces[0], eyes[0])
    else:
        if (oldface is None or oldeyes is None):
    	    return (None, None)
        else:
            return (oldface, oldeyes)


oldleft=None
oldright=None
def findtheeyes(img, eye_pair):
    global oldleft, oldright    # Cached result from previous run

    # Find the left and right eye separately. (Not very robust).
    # This often hallucinates eyes or can't see them. Also, the
    # region it finds is not as tight as eyepair_big's rectangle.
    # In particular, it extends too high, up to the brows.
    #
    # To handle this, we intersect with the eyepair rectangle to
    # find an eye that's on the proper side of the face.

    roi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lefteyes = lefteye_cascade.detectMultiScale(roi_gray)
    left = None
    for (lx,ly,lw,lh) in lefteyes:
        # Check each eye found and see if it is on the left side
        # of the eye pair rectangle. 
        left = (lx+lw/2, ly+lh/2) 	# Center of eye
        if not isWithinLeft( left, eye_pair ):
            # Reject 
            exOut(img, (lx,ly,lw,lh), magenta, 1)
            left = None
            continue
        else:
            # Accept
            cv2.rectangle(img,(lx,ly,lw,lh),magenta,2)
            break

    righteyes = righteye_cascade.detectMultiScale(roi_gray)
    right = None
    for (lx,ly,lw,lh) in righteyes:
        # Check each eye found and see if it is on the right side
        # of the eye pair rectangle. 
        right = (lx+lw/2, ly+lh/2) 	# Center point of eye
        if not isWithinRight( right, eye_pair ):
            # Reject
            exOut(img, (lx,ly,lw,lh), yellow, 1)
            right = None
            continue
        else:
            # Accept
            cv2.rectangle(img,(lx,ly,lw,lh),yellow,2)
            break

    if (left and right):
        (oldleft, oldright) = (left, right)
        if (abs(left[0]-right[0])<20):
            print( "Left", left, "\tRight", right, "\tEye pair", eye_pair )
            print( "Left  is within eye_pair:", isWithinLeft( left,  eye_pair) )
            print( "Right is within eye_pair:", isWithinRight(right, eye_pair) )

    return (oldleft, oldright)


def roi(img, rect):
    """ 
    Given an image and a rectangle, return the 2-D slice of that image.
    NOTE: Writing to the array returned will alter the original image. 
    """
    (x,y,w,h) = rect
    return img[y:y+h, x:x+w]

def centereyesscalechin(img, x_y_w_h, ex_ey_ew_eh):
    x,y,w,h = x_y_w_h
    ex,ey,ew,eh = ex_ey_ew_eh

    # Given the image and bounding boxes for the face and eyes,
    # return a transform matrix that will 
    # recenter the image and scale it so the eyes are in the middle and
    # the chin is the proper proportion from the bottom.

    height, width, channels = img.shape

    heightofchin=y+h                      # Bottom of bounding box.
    heightofeyes=((y+ey) + (y+ey+eh))/2.0 # Eyes are relative to face box
    chintoeyes=abs(heightofchin - heightofeyes)

    # The eyes are in the middle and there is a chin_height
    # percentage gap between chin and bottom of picture. (E.g.,
    # 1/7th). That means, the distance from the chin to the eyes,
    # chintoeyes, should be, once scaled, be equal to
    # (1/2- chin_height) times the image height. (E.g., 5/14th).

    scale = (0.5-chin_height) * height/chintoeyes

    # Scale all measurements
    (x, y, w, h, ex, ey, ew, eh) = np.dot (
            (x, y, w, h, ex, ey, ew, eh), scale )

    # Translation needed in the X and Y directions to put eyes in center
    tx = width/2-(x+ex+(x+ex+ew))/2
    ty = height/2-(y+ey+(y+ey+eh))/2

    # The transformation matrix is the scaling + translation.
    M = np.float32([[scale,0,tx],[0,scale,ty]])
    return M


def buggyiodtransform(img, l_r, r=None):
    # Intraocular distance transform, same as below, but buggy 

    if r is not None:
        l = l_r
    else:
        (l, r) = l_r
    
    (Lu, Lv) = l
    (Ru, Rv) = r
    (h, w, channels) = img.shape

    Hyp = sqrt( (Ru-Lu)**2  +  (Rv-Lv)**2 )
    IOD = eye_distance
    scale = w*IOD/(Ru-Lu)
    M = np.float32(
        [ [  (Ru-Lu)/Hyp*scale,  (Rv-Lv)/Hyp,  w/2 - w*IOD/2 - Lu*scale ],
          [ -(Rv-Lv)/Hyp,  (Ru-Lu)/Hyp*scale,  h - eye_height*h - Lv*scale ] ] )

    return M


def iodtransform(img, l_r, r=None):
    # Intraocular distance transform.
    #
    # Given an image and the center of left and right eyes,
    # return the image warped such that eyes are:
    # * horizontal,
    # * at the correct eye_height, and
    # * the proper distance apart (eye_distance).
    #
    # Note: eye_height and _distance are fractions of image height/width. 

    if r is not None:
        l = l_r
    else:
        (l, r) = l_r
    
    (Lu, Lv) = l
    (Ru, Rv) = r
    (h, w, channels) = img.shape

    Hyp = sqrt( (Ru-Lu)**2  +  (Rv-Lv)**2 )
    IOD = eye_distance

    F = np.float32( [ [ 1, 0, -Lu ],
                      [ 0, 1, -Lv ],
                      [ 0, 0,  1  ] ] )

    G = np.float32( [ [  (Ru-Lu)/Hyp,  (Rv-Lv)/Hyp, 0 ],
                      [ -(Rv-Lv)/Hyp,  (Ru-Lu)/Hyp, 0 ],
                      [ 0,       0,      1 ] ] )

    scale=w*IOD/(Ru-Lu)
    H = np.float32( [ [ scale, 0,        0 ],
                      [ 0,        scale, 0 ],
                      [ 0,        0,        1 ] ] )

    J = np.float32( [ [ 1, 	0, 	w/2 - w*IOD/2       ],
                      [ 0, 	1, 	h - eye_height*h    ],
                      [ 0, 	0, 	1                   ] ] )

    Fp = np.float32( [ [ 1, 0, Lu ],
                       [ 0, 1, Lv ],
                       [ 0, 0,  1 ] ] )

    Gp = np.float32( [ [  (Ru-Lu)/Hyp, -(Rv-Lv)/Hyp, 0 ],
                       [  (Rv-Lv)/Hyp,  (Ru-Lu)/Hyp, 0 ],
                       [ 0,       0,      1 ] ] )

    Hp = np.float32( [ [ Hyp/(w*IOD),      0,          0 ],
                       [ 0,           Hyp/(w*IOD),     0 ],
                       [ 0,               0,           1 ] ] )

    Jp = np.float32( [ [ 1, 	0,  	w*IOD/2 - w/2   ],
                       [ 0, 	1, 	eye_height*h    ],
                       [ 0, 	0,  	1               ] ] )

    # print( "F\n", F )
    # print( "G\n", G )
    # print( "H\n", H )
    # print( "J\n", J )
              
    # img = cv2.warpAffine( img, F[:][0:2], img.shape[1::-1] )
    # img = cv2.warpAffine( img, G[:][0:2], img.shape[1::-1] )
    # img = cv2.warpAffine( img, H[:][0:2], img.shape[1::-1] )
    # img = cv2.warpAffine( img, J[:][0:2], img.shape[1::-1] )

    # Apply all the transformations together as a single transform using matrix multiply
    return (J @ H @ G @ F)[:][0:2]

def crop(img):
    # Given an image, and the global variable photo_aspect,
    # return image cropped to correct aspect ratio, centered on the original.
    height, width, channels = img.shape
    (final_width, final_height) = maxpect(photo_aspect, width, height)
    tx=(final_width-width)/2
    ty=(final_height-height)/2
    M = np.float32([[1,0,tx],[0,1,ty]])
    img = cv2.warpAffine(img,M,(final_width,final_height))
    return img

def recalculate_frame_downscale(downscale):
    # Sets frame_downscale based on frame_width, frame_height, and downscale.

    # FRAME_WIDTH, FRAME_HEIGHT is the size of the video capture frame.
    # DOWNSCALE is the desired maximum width or height.

    # FRAME_DOWNSCALE is the (width, height) for Haar processing and display.
    # It retains the aspect ratio of the video capture frame
    # (frame_width : frame_height), but fits in a square of length downscale.
    # E.g., 

    # When downscale is zero, frame_downscale is simply the frame geometry.

    global frame_downscale
    global frame_width, frame_height
    global oldface, oldeyes

    if downscale == 0:
        frame_downscale = (int(frame_width), int(frame_height))
        print("Downscaling for internal processing disabled. Using %d x %d." % frame_downscale)
    else:
        frame_downscale = maxpect(frame_width/frame_height, downscale, downscale)
        print("Downscaled size for internal processing is now %d x %d."
              % frame_downscale)

    # Invalidate found face. XXX should probably just rescale them.
    oldface = None
    oldeyes = None

    return 

def getWindowSize ( title ):
    """OpenCV has no way to check the current windowsize except by asking
    for a rectangle, which bizarrely sometimes fails.
    """
    (dummy,dummy,width,height) = cv2.getWindowImageRect(title)
    if (width == -1 or height == -1):
        eprint("getWindowSize Error -1, -1")
    return (width, height)

def isWithin(point, rect_x_y_w_h, ry=None, rw=None, rh=None):
    """Return true if a point is within a rectangle, inclusive"""
    px,py = point
    if not ry:
        (rx,ry,rw,rh) = rect_x_y_w_h
    else:
        rx = rect_x_y_w_h
    return (rx <= px and px <= rx+rw and  ry <= py and py <= ry+rh)

def isWithinLeft(point, rect_x_y_w_h, ry=None, rw=None, rh=None):
    """Return true if a point is within left quarter of a rectangle"""
    if not ry:
        (rx,ry,rw,rh) = rect_x_y_w_h
    else:
        rx = rect_x_y_w_h
    return isWithin( point, rx, ry, rw/4, rh )

def isWithinRight(point, rect_x_y_w_h, ry=None, rw=None, rh=None):
    """Return true if a point is within right quarter of a rectangle"""
    if not ry:
        (rx,ry,rw,rh) = rect_x_y_w_h
    else:
        rx = rect_x_y_w_h
    return isWithin( point, rx+3*rw/4, ry, rw/4, rh )


def exOut(img, rect_x_y_w_h, color_y, thickness_w, h=None, color=None, thickness=None):
    if not h:
        (x,y,w,h) = rect_x_y_w_h
        color = color_y
        thickness = thickness_w
    else:
        (x,y,w,h) = (rect_x_y_w_h, color_y, thickness_w, h)
        
    cv2.line(img,(x,y),(x+w,y+h),color,thickness)
    cv2.line(img,(x+w,y),(x,y+h),color,thickness)

    
def get_fourcc(cap: cv2.VideoCapture) -> str:
    """Return the 4-letter string of the codec the camera is using.

    'cap' is an OpenCV camera opened using VideoCapture().

    If the VideoCapture device is not opened or if the backend is not
    capable, a string of four nulls will be returned. ('\\x00\\x00\\x00\\x00')

    NB: as of 2023, the GSTREAMER backend always fails. Use V4L2 instead.
    """
    return int(cap.get(cv2.CAP_PROP_FOURCC))\
             .to_bytes(4, byteorder=sys.byteorder).decode()

def set_fourcc(cap: cv2.VideoCapture, codec: str) -> bool:
    """Set the the codec the camera is using for capture.

    'cap' is an OpenCV camera opened using VideoCapture().

    'codec' is a fourcc string such as 'YUYV' or 'MJPG'.

    Return value is True iff the camera codec was successfully changed.

    NOTA BENE: As of 2023, the GSTREAMER backend always fails.
    Open your camera using cv2.VideoCapture(0, cv2.CAP_V4L2) instead.

    Note: As of 2023, OpenCV has no way to enumerate available codecs.
    Under GNU/Linux, you can run 'v4l2-ctl --list-formats-ext' from the
    command line.
    """
    cap.set(cv2.CAP_PROP_FOURCC,
            int.from_bytes(codec.encode(), byteorder=sys.byteorder))
    return codec == get_fourcc(cap)

def print_debug_info(face, eyes, left, right):
    print()
    print(f"face is {face}\t left eye is {left}")
    print(f"eye pair is {eyes}\tright eye is {right}")
    print("GUI props")
    print("\trectangle", cv2.getWindowImageRect(title))
    print("\tfullscreen: ",cv2.getWindowProperty(title, cv2.WND_PROP_FULLSCREEN))
    print("\tautosize: ",cv2.getWindowProperty(title, cv2.WND_PROP_AUTOSIZE))
    print("\taspect ratio: ",cv2.getWindowProperty(title, cv2.WND_PROP_ASPECT_RATIO))
    print("\topengl: ",cv2.getWindowProperty(title, cv2.WND_PROP_OPENGL))
    print("\tvisible: ",cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE))
    print("\ttopmost: ",cv2.getWindowProperty(title, cv2.WND_PROP_TOPMOST))
    print("Camera is capturing at %d x %d at %d FPS." %
          (frame_width, frame_height, capture.get(cv2.CAP_PROP_FPS) ))

    print("Output image size will be %d x %d" %
          maxpect(photo_aspect, frame_width, frame_height))


def print_help():
    print("""
Keys available:
    q  (or Esc)		Quit
    Spacebar		Save image to passport.jpg
    f			Toggle fullscreen
    r			Change camera rotation by 90 degrees
    m			Toggle mirroring of image
    1-5			Change internal downscaling (for slow computers)
    0			Disable downscaling completely
    Tab			Switch scaling method (eye distance or chin)
    Enter		Show debugging information
    """)


def main():    
    global downscale, frame_downscale, frame_height, frame_width
    global camera_mirrored, camera_rotation
    global face_warp, use_chin_scaling 

    # OpenCV coordinates of detected features 
    face = None                 # Rectangle around the face (x,y,w,h)
    eyes = None                 # Rectangle around the pair of eye (x,y,w,h)
    left = None                 # Point at center of left eye (x,y)
    right = None                # Point at center of right eye (x,y)

    init()

    while True:
        (rv, original) = capture.read()
        if (not rv): break

        fps.incrementFrames()

        # Rotate the image, if the camera is on its side
        if (camera_rotation):
            original=np.rot90(original, camera_rotation)

        # Downscale image to make findtheface() faster
        if downscale:
            img = cv2.resize(original, frame_downscale)
        else:
            img = original.copy()

        # Find the face and a pair of eyes using the Haar cascade
        (face, eyes) = findtheface(img)

        if (face is not None):
            roi_color = roi(img, face)

            if (eyes is not None):
                # Find the left and right eyes using the Haar cascade
                (left, right) = findtheeyes(roi_color, eyes)

                if (left is not None and right is not None):
                    # Make left and right eye position absolute
                    # instead of relative to the face.
                    left = left + face[0:2]
                    right = right + face[0:2]

                    if not use_chin_scaling:
                        face_warp = iodtransform(img, left, right)
                    else:
                        # Fallback to using the chin distance for scaling.
                        face_warp = centereyesscalechin(img, face, eyes)

                    img = cv2.warpAffine( img, face_warp, img.shape[1::-1] )

                    # Crop to the proper aspect ratio
                    img=crop(img)

                    # OpenCV's resizeWindow is buggy, especially with fullscreen
#                    cv2.resizeWindow(title, img.shape[0:2])


        # Show the image (and frames per second)
        cv2.imshow(title, cv2.flip(img,1) if camera_mirrored else img)

        if (fps.framecount % 10 == 0):
            eprint('%.2f fps' % fps.getFPS(), end='\r')
            fps.reset()

        # Show image and wait for a key
        k = cv2.waitKey(1)
        c = chr(k & 0xFF)

        if (k == -1):           # No key hit, wait timed out.
            continue

        elif (c == 'q' or c == '\x1b'): 	# q or ESC to quit
            break

        elif (c == ' ' or c=='p' or c=='s'):    # Save a screenshot
            if (face_warp is not None):
                scale = original.shape[0] / img.shape[0]
                M = face_warp
                M[0][2] *= scale # Scale up the translation part of the transform.
                M[1][2] *= scale
                img = cv2.warpAffine( original, M, original.shape[1::-1] )
                img = crop(img)
                cv2.imwrite("passport.jpg", img)
                print("Wrote image to passport.jpg")
            else:
                eprint('Error: No face detected yet!\a')

        elif (ord('0') <= ord(c) and ord(c) <= ord('9')  ): 
            # Try lower res for speed
            if   (c == '1'): downscale = 160
            elif (c == '2'): downscale = 320
            elif (c == '3'): downscale = 640
            elif (c == '4'): downscale = 960
            elif (c == '5'): downscale = 1280
            else:            downscale = 0

            recalculate_frame_downscale(downscale)
            cv2.resizeWindow( title, frame_downscale )

        elif (c == 'f'):          		# Toggle full screen
            isFull = cv2.getWindowProperty(title, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, 1 - isFull)

        elif (c == 'r'):          		# Rotate camera
            camera_rotation = (camera_rotation + 1) % 4
            frame_width, frame_height = frame_height, frame_width
            recalculate_frame_downscale(downscale)

        elif (c == 'm'):          		# Toggle mirroring
            camera_mirrored = 1 - camera_mirrored

        elif (c == '\t'):          		# Switch US/CN photo standard
            use_chin_scaling = 1 - use_chin_scaling
            print( "US Passport" if use_chin_scaling == 0 else "Chinese Visa",
                   "scaling algorithm now in use." )

        elif (c == '\r' or c == '*'):        # debugging
            print_debug_info(face, eyes, left, right)

        elif (c == '?' or c == 'h'):          		# show help
            print_help()
        else:
            eprint('Unknown key %c (%x)' % (c, k))

    # End of main loop
    capture.release()
    cv2.destroyAllWindows()


main()
