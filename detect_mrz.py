# -*- coding: utf-8 -*-
# author: haroldchen0414

import numpy as np
import cv2

imagePath = "test.jpg"
image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), -1)
image = cv2.resize(image, (600, 600), cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blured = cv2.GaussianBlur(gray, (3, 3), 0)
# balckhat操作用于在浅色背景凸显黑色区域，即护照上的字体
blackHat = cv2.morphologyEx(blured, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)))
#cv2.imshow("BlackHat", blackHat)
#cv2.waitKey(0)

# 计算blackhat图像沿x轴的scharr梯度并且缩放回[0, 255]范围
gradX = cv2.Sobel(blackHat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))

gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)))

# 减小字符之间的间隙，用Otsu方法进行阈值处理
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))
thresh = cv2.erode(thresh, None, iterations=5)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    # 计算纵横比，纵横比很大的区域可以认为是机读区域
    ar = w / float(h)

    if ar > 10:
        pX = int((x + w) * 0.03)
        pY = int((y + h) * 0.03)
        (x, y) = (x - pX, y - pY)
        (w, h) = (w + (pX * 2), h + (pY * 2))

        roi = image[y: y+h, x: x+w].copy()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imshow("ROI", roi)
cv2.waitKey(0)


