import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

template = cv2.imread('./data/2.jpg')
th, tw, _ = template.shape
image = cv2.imread('./data/3.jpg')
# image = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5)
h, w, _ = image.shape
scale = h / th

template = cv2.resize(template, dsize=(0, 0), fx=scale, fy=scale)
found = None

for scale in np.linspace(0.3, 1, 100)[::-1]:
    resized_template = imutils.resize(template, width=int(template.shape[1] * scale))
    r = 1
    result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
    (_, maxval, _, maxloc) = cv2.minMaxLoc(result)

    if found is None or maxval > found[0]:
        print(maxval)
        found = (maxval, maxloc, r)
        th, tw, _ = resized_template.shape

(_, maxloc, r) = found
(startX, startY) = (int(maxloc[0] * r), int(maxloc[1] * r))
(endX, endY) = (int((maxloc[0] + tw) * r), int((maxloc[1] + th) * r))

# 在检测结果上绘制边界框并展示图像
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
