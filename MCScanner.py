import cv2
import numpy as np

photo_img = cv2.imread('photo.jpg')
scan_img = cv2.imread('scan.jpg')

grayscaled = cv2.cvtColor(scan_img,cv2.COLOR_BGR2GRAY)
gaussian = cv2.adaptiveThreshold(grayscaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,151,20)

gaussian = cv2.resize(gaussian, (0,0), fx=0.5, fy=0.5)


cv2.imshow('gaussian', gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()