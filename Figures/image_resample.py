import os
import cv2

path="C:/Users/ali.polat/‪Desktop/q"
x=0
for i in os.listdir(path):

    if i.startswith("fig"):
        x=x+1
        image=os.path.join(path,i)
        image=cv2.imread(image)
        h,w,_=image.shape
        oran=float(1024.0/w)

        new_w=w*oran
        new_h=h*oran
        dim=(int(new_w),int(new_h))
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        name="Figure"+str(x)+".jpg"
        cv2.imwrite(os.path.join(name),resized)
        print h,w
        print i, "yeni boy_en:",new_w,new_h

        