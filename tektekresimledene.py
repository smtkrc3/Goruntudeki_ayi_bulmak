import cv2
import numpy as np

ay=cv2.imread("resim8.jpg")

ay_gri=cv2.cvtColor(ay,cv2.COLOR_BGR2GRAY)  # gray renk uzayı,  gri hale cevirip 3 kanalı teke düşürdüm https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

ay_blur = cv2.GaussianBlur(ay_gri,(5,5),0)  # resmi bulanıklaştırarak noise (gürültü) detayların azaltılması için kullandım https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

ret, ay_threshold=cv2.threshold(ay_blur, 85, 255,cv2.THRESH_BINARY) # değerleri keyfime göre verdim
# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html farklı türleri de var

ay_Mask=cv2.bitwise_and(ay,ay,mask=ay_threshold) # ay_Mask BGR space color çünkü kaynağı ay'dan aldık
# bitwise_and ile ana resimdeki renkli yerleri thresholddaki beyazlara ekledi(thresholddaki beyaz siyah durumu ana fotografla karışımda sınır noktası görevinde)
# Neden beyazlara ekledi? bitwise_and (1 ve 1 olunca calıstı) thresholddaki 1 olan beyaz değerleri ile ana fotoğraf eşleşti.
ay_HSV=cv2.cvtColor(ay_Mask,cv2.COLOR_BGR2HSV)

gri_alt_sinir=np.array([0,0,15])  # HSV sınırları (Hue,Saturation,Value)=(Renk,Doygunluk,Parlaklık)
gri_ust_sinir=np.array([225,72,255])     # 10 fotoğraf üzerinde test edip bu sayilarda karar kıldım
# kodlaması bitince yeni 10 adet fotoğraf daha ekleyip yüzde kaç hata payı var onu hesaplayacağım

gri_renk_filtresisonucu= cv2.inRange(ay_HSV,gri_alt_sinir,gri_ust_sinir)

#cv2.imshow("ay gri",ay_gri)
#cv2.imshow("ay blur",ay_blur)
#cv2.imshow("ay threshold",ay_threshold)
#cv2.imshow("ay mask",ay_Mask)
#cv2.imshow("ay hsv",ay_HSV)
#cv2.imshow("Filtre",gri_renk_filtresisonucu)

#Contours
# better accuracy -> binary image (0 1) Gray color space sanırım
# threshold  edge detection
# https://www.youtube.com/watch?v=FbR9Xr0TVdY     https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
# In OpenCV, finding contours is like finding white object from black background. So remember,
# object to be found should be white and background should be black.

Deneme=ay.copy()     #kopyalama amacım orijinal resmin değişmemesi (eğer kopyalamazsam pointer varmış gibi çalışır)    
contours, _ = cv2.findContours(gri_renk_filtresisonucu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Bana en büyük contour lazım       https://www.youtube.com/watch?v=-LlaxL5iE8U&t=2351s burada anlatılıyor
c = max(contours, key = cv2.contourArea)    # internetten hazır aldım https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(Deneme,(x,y),(x+w,y+h),(0,255,0),2)


#cv2.drawContours(Deneme, contours, -1, (0,255,0), 3)
cv2.imshow("Kare Icinde Ay",Deneme)


cv2.waitKey(0)
cv2.destroyAllWindows()
