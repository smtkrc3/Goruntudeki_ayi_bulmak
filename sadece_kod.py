import cv2
import numpy as np

ay=cv2.imread("resim6.jpg")

ay_gri=cv2.cvtColor(ay,cv2.COLOR_BGR2GRAY) 

ay_blur = cv2.GaussianBlur(ay_gri,(5,5),0)   

ret, ay_threshold=cv2.threshold(ay_blur, 85, 255,cv2.THRESH_BINARY) 

ay_Mask=cv2.bitwise_and(ay,ay,mask=ay_threshold) 
ay_HSV=cv2.cvtColor(ay_Mask,cv2.COLOR_BGR2HSV)

gri_alt_sinir=np.array([0,0,15])
gri_ust_sinir=np.array([225,72,255])

gri_renk_filtresisonucu= cv2.inRange(ay_HSV,gri_alt_sinir,gri_ust_sinir)

Deneme=ay.copy()  
contours, _ = cv2.findContours(gri_renk_filtresisonucu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key = cv2.contourArea)    
x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(Deneme,(x,y),(x+w,y+h),(0,255,0),2)

#cv2.drawContours(Deneme, contours, -1, (0,255,0), 3)
cv2.imshow("Kare Icinde Ay",Deneme)


cv2.waitKey(0)
cv2.destroyAllWindows()