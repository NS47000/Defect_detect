import cv2
import numpy as np

width=1440
height=1080

subpixel=40
for dot_pixel in range(1,5):
    img=np.zeros((height,width),dtype=np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            if j%subpixel==0 and i%(subpixel*2)==0:
                img[i,j]=255
                if dot_pixel>1:
                    img[i+1,j]=255
                if dot_pixel>2:
                    img[i,j+1]=255
                if dot_pixel>3:
                    img[i+1,j+1]=255
            if j%subpixel==subpixel//2 and i%(subpixel*2)==subpixel:
                img[i,j]=255
                if dot_pixel>1:
                    img[i+1,j]=255
                if dot_pixel>2:
                    img[i,j+1]=255
                if dot_pixel>3:
                    img[i+1,j+1]=255

    #img=255-img
    cv2.imwrite("defect_pattern_{}_white.png".format(dot_pixel),img)
    #img=cv2.imread("test.png",0)
    #img=255-img
    #cv2.imwrite("test_white.png",img)
    
img=np.zeros((height,width),dtype=np.uint8)
white=255-img
cv2.imwrite("white.png",white)
for i in range(0,width):
    if i//3%2==0:
        img[:,i]=255
cv2.imwrite("mtf.png",img)
img=np.zeros((height,width),dtype=np.uint8)

img=img[540-]
    