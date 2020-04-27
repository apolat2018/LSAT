# -*- coding: utf-8 -*-
import os
import cv2


image=cv2.imread("‪C:/Users/AP/Documents/GitHub/LSAT/figure5.jpg")
w,h,_=image.shape
oran=float(1024.0/w)
new_w=w*oran
new_h=h*oran
dim=(int(new_w),int(new_h))
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

cv2.imwrite("‪C:/Users/AP/Documents/GitHub/LSAT/figure5_1.jpg",resized)