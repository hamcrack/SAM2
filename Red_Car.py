import cv2
import urllib.request
import numpy as np

def img_from_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img
    
img_1_url = 'https://play-lh.googleusercontent.com/XGvcA4Xu34hh3iTT3Jrl1l1jijtZHqsVtYpzdfcuJNJcuZoPku7Wq7LIHHdVm-soLALH'
image = img_from_url(img_1_url)
while(1):
    cv2.imshow('THEKER image 1', image)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue, sat, val = cv2.split(hsv)
hue[hue > 3] = 255
hue[hue <= 3] = 0
hue = cv2.bitwise_not(hue)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hue, 4, cv2.CV_32S)
for i in range(num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    # print(i, ": ", area)
    if 1000 < area < 10000:
        (cX, cY) = centroids[i]
        component = image[y:(y + h), x:(x + w), :]
        # image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

while(1):
    cv2.imshow('Red Car', component)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()