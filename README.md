# passportpix

Take a passport photo of precisely the proper dimensions using
computer vision to pan and zoom on the face.

The program works, but does require you to edit a line by hand at the
top if you want to change the image width-to-height ratio.

Shows live camera view. Automatically centers and zooms image to be
precisely correct for passport photos. Hit spacebar to snap the photo
to "passport.jpg" and <kbd>q</kbd> to quit.

## NOTE

Preview on screen is lower resolution than saved to a file. The
face detection routine is a bit CPU hungry so the image is downscaled
to fit in a 640x640 square before processing to get a better frame
rate. Likewise, some low-end computers, such as the Raspberry Pi,
struggle with displaying large images rapidly, so the downscaled image
is used for screen display.

# TIP

If you leave an image viewer, such as `eog passport.jpg`, running
in the background, you'll see the picture update immediately so you
can tell if you like it or not.

# TIP

If you need a "portrait" photo (taller than it is wide), you can
improve the image resolution by rotating your camera sideways and edit
the `camera_rotation` variable at the top of the file. 

## Customization

At the moment, size is hardcoded to 33mm x 48mm, with the distance
from the chin to the bottom of the photo being 7mm. However, it's
pretty darn easy to change by editing some numbers at the top of the
file. For example, for a US Passport, you could set `image_ratio =
2.0/2.0`.

## Current keys

* <kbd>Space</kbd>: Save snapshot to passport.jpg  
* <kbd>q</kbd> or <kbd>Esc</kbd>: Quit

### Experimental keys for resolution

These keys change the downscale resolution which is used for both the
computer vision processing and for display on the screen. They do not
affect the output resolution in the saved file.

| Key          | Maximum height or width |
|--------------|-------------------------|
| <kbd>1</kbd> | 160                     |
| <kbd>2</kbd> | 320                     |
| <kbd>3</kbd> | 640 (default)           |
| <kbd>4</kbd> | 960                     |
| <kbd>5</kbd> | 1280                    |
| <kbd>0</kbd> | Native resolution       |

Note that OpenCV's builtin face detection algorithms failed for me on
160×160 images.


## Current Assumptions

* I'm assuming you will always want your eyes precisely centered in the
picture.

* I assume you always want the highest resolution possible from your camera.

* You'll need to have 'python-opencv' installed. (`apt install python-opencv`) 

## Bugs and Future Features.

* OpenCV appears to prefer YUYV (uncompressed) video, even if a camera
  can provide a higher resolution & frame rate with MJPG (compressed).
  This should only be a problem for older USB 2 cameras. 

* OpenCV cannot write the proper DPI to the JPEG file, so it will not
  print out at the correct size without massaging. This is annoying, but
  is easy enough to work around that I'm not fixing it before the
  release.

* You can only take one photo. Every photo you take overwrites the
  previous 'passport.jpg' file.

* Only the first camera is opened. You can change that by editing
  `camera_device` at the beginning of the file.

* When recentering the face, especially if it is very close to the
  camera, there may not be enough picture from the camera to extend all
  the way to the edge of the final photo. The program should warn about
  this and give a visual indication of which way to move to fix it.

* If your face is too far away, the camera will zoom in too much,
  causing a blurry picture. This program ought to detect if height or
  width is less than 600 and ask the user to step closer.

## Debugging & such

Mostly reminders to myself.

* To list all resolutions a camera is capable of:

  ```
  v4l2-ctl --list-formats-ext
  ```

* To print out the current GUI properties, hit <kbd>*</kbd>.

* OpenCV has unnecessarily confusing GUI window properties. In
  particular, it appears OpenCV was originally written using simple
  integer Booleans (0 or 1), but someone later came along and decided
  that was too sloppy and renamed them all. However, instead of using
  the typical True or False, they came up with new names for each
  variable so each has its own nearly-unique way of being used.

  I found this silly and hard to read, so I do not follow that
  practice. For example, I have replaced the following code:

  ```python
  isFull = (cv2.getWindowProperty(title, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN)a
  cv2.setWindowProperty(title,
			cv2.WND_PROP_FULLSCREEN,
			cv2.WINDOW_NORMAL if isFull else cv2.WINDOW_FULLSCREEN)
  ```

  with:

  ```python
  isFull = cv2.getWindowProperty(title, cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, 1 - isFull)
  ```

* Here are the OpenCV window properties and their official
  documentation (as of 2023). _(Italics mine.)_

   * `WND_PROP_FULLSCREEN`<br/>
       fullscreen property (can be `WINDOW_NORMAL` or `WINDOW_FULLSCREEN`).<br/>
       _Boolean `NORMAL` is 0 and `FULLSCREEN` is 1_

   * `WND_PROP_AUTOSIZE`<br/>
       autosize property (can be `WINDOW_NORMAL`, 0, or `WINDOW_AUTOSIZE`, 1).<br/>
       _Boolean `NORMAL` is 0 and `AUTOSIZE` is 1_
	   
   * `WND_PROP_ASPECT_RATIO`<br/>
       window's aspect ration (can be `WINDOW_FREERATIO` or `WINDOW_KEEPRATIO`).<br/>
       _Boolean `KEEPRATIO` is 0 and `FREERATIO` is 256_
	   
   * `WND_PROP_OPENGL`<br/>
       opengl support.<br/>
       _Presumed to be Boolean, but documentation does not specify._
	   
   * `WND_PROP_VISIBLE`<br/>
       checks whether the window exists and is visible.<br/>
       _Presumed to be Boolean, but documentation does not specify._
	   
   * `WND_PROP_TOPMOST`<br/>
       property to toggle normal window being topmost or not. <br/>
       _Presumed to be Boolean, but documentation does not specify._

## Bonus: Harpy.py

Here is a simple Python script for face detection in live video using
OpenCV. <a href="harpy.py">harpy.py</a>. Short and easy to modify if
you want to start another project like this.

<img src="./README.md.d/harpy.py-screenshot2.jpg">

<img src="./README.md.d/harpy.py-screenshot3.jpg">



