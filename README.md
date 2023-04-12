# passportpix

Take a passport photo of precisely the proper dimensions using
computer vision to pan and zoom on the face.

Defaults to US Passport requirements, but can be modified for others.

Shows live camera view. Automatically centers and zooms image to be
precisely correct for passport photos. Hit spacebar to snap the photo
to "passport.jpg" and <kbd>q</kbd> to quit.

## How to run

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

The default setup is correct for US Passport photos:

| Variable     | Description                                                              | Setting for US Passport         |
|--------------|--------------------------------------------------------------------------|---------------------------------|
| photo_aspect | Ratio of width to height                                                 | 1<br/>(square)                  |
| eye_distance | Distance between eyes, expressed as a fraction of the picture width      | 2/12<br/>(= ⅓" on a 2" width)   |
| eye_height   | Distance of eyes from bottom of picture, as a fraction of picture height | 7/12<br/>(= 1⅙" on a 2" height) |

These variables are listed at the top of passport.py and can be
changed there.

### Example customization

If one wanted to get a visa to visit China, the requirements as of
2020 are:

| Variable     | Description                                                              | Setting for CN Visa             |
|--------------|--------------------------------------------------------------------------|---------------------------------|
| photo_width  | Width of printed photo                                                   | 33                              |
| photo_height | Height of printed photo                                                  | 48                              |
| photo_units  | Unit of measurement of width and height                                  | mm                              |
| photo_aspect | Ratio of width to height                                                 | 33/48                           |
| eye_height   | Distance of eyes from bottom of picture, as a fraction of picture height | 24/48<br/>(precisely in middle) |
| chin_height  | Distance of chin from bottom of picture, as a fraction of picture height | 7/48                            |

So, one would set:

``` python
# Setting for Chinese Visa photo
photo_width  = 33.0
photo_height = 48.0
photo_units  = "mm"
photo_aspect = photo_width/photo_height
eye_height   = 24.0 / 48.0
chin_height  = 7.0 / 48.0
```

## Current keys

* <kbd>Space</kbd>: Save snapshot to passport.jpg  
* <kbd>f</kbd>: Toggle fullscreen  
* <kbd>m</kbd>: Toggle mirroring  
* <kbd>q</kbd> or <kbd>Esc</kbd>: Quit

### Experimental keys for resolution
<details><summary><b>Click to see more keys</b></summary>
<blockquote>

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
</blockquote>

## Current Assumptions

* I'm assuming you will always want your eyes precisely centered in the
picture.

* I assume you always want the highest resolution possible from your camera.

* You'll need to have 'python-opencv' installed. (`apt install python-opencv`) 

## Bugs and Future Features.

* Cropped image jitters as eye locations are approximated. Ought to
  show uncropped camera view either in a second window or alone with a
  rectangle showing how the image will be cropped.

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

## Debugging &amp; such
<details><summary><b>Click to see info about debugging</b></summary>
<blockquote>

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

</blockquote>
</details>

## Bonus: Harpy.py

Here is a simple Python script for face detection in live video using
OpenCV. <a href="harpy.py">harpy.py</a>. Short and easy to modify if
you want to start another project like this.

<img src="./README.md.d/harpy.py-screenshot2.jpg">

<img src="./README.md.d/harpy.py-screenshot3.jpg">



