#!/usr/bin/python3
import pandas as pd
import cv2

url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
table = pd.read_html(url)[0]
table.columns = table.columns.droplevel()
backend = cv2.CAP_V4L2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if cap.isOpened() is False:
    print('Could not open camera with driver %s' %
          cap.getBackendName())
    exit(1)
resolutions = {}
for index, row in table[["W", "H"]].iterrows():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resolutions[str(width)+"x"+str(height)] = "OK"
print(resolutions)
