import numpy as np
import cv2
from skimage import data, io, filter, measure
from matplotlib import pyplot as plt
from skimage import morphology
from skimage.draw import polygon
from scipy.signal import argrelmax
import numpy as np
import cv2
from pd import peakdet
"""
# Create a black image
#img = cv2.imread('samolot01.jpg',0)
#img = np.zeros((512,512,3), np.uint8)

#ero = morphology.erosion(img, morphology.square(5))
#sob = filter.sobel(ero)
#sobc= cv2.Sobel(imgray, ddepth = -1,dx = 1,dy = 1)
im = cv2.imread('diff.jpg')
im2 = cv2.imread('diff1.jpg')
#im3 = cv2.compare(im,im2, cmpop=cv2.CMP_EQ)
im3 = cv2.absdiff(im,im2)
#im=im>0.45
imgray = cv2.cvtColor(im3,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,10,255,0)
#ero = cv2.erode(thresh, kernel=None, iterations=3 )
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#f = open('con.txt','w')
#f.write(string(contours))
#------------------------------------------------------  TYLKO NAJDLUZSZY KONTUR
if len(contours)>0:
    maxi=0
    i=1
    while i!=len(contours):
        if len(contours[i])>len(contours[maxi]):
            maxi=i
        i=i+1
#print len(contours[maxi])
cv2.drawContours(im2,[contours[maxi]],-1,(0,255,0),3)
#------------------------------------------------------ KONWERSJA DANYCH Z FIND CONTOURS DO OSI X / Y
xaxis=[]
yaxis=[]
for i in range(len(contours[maxi])):
    xaxis.append( (((contours[maxi])[i])[0])[0])
    yaxis.append( (((contours[maxi])[i])[0])[1])
#------------------------------------------------------ CENTROID
x=(max(xaxis) + min(xaxis))/2
y=(max(yaxis) + min(yaxis))/2
cv2.circle(im2,(x,y), 3, (0,0,255), -1)
#------------------------------------------------------ ODLEGLOSCI WSZYSTKICH PUNKTOW OD CENTROIDA
odl=[]
for i in range(len(xaxis)):
    odl.append(  np.sqrt((x-xaxis[i])**2 + (y-yaxis[i])**2  )   )
#------------------------------------------------------ DETEKCJA EKSTREMOW
#odl1 = np.array(odl)
#sss=argrelmax(odl1)
maxtab, mintab = peakdet(odl,30.0)

#------------------------------------------------------ WYCIAGANIE POTRZEBNYCH DANYCH Z WYNIKU PEAKDET
sk=[]
for i in (maxtab[:,0]):
    sk.append(int(i))
#------------------------------------------------------ RYSOWANIE EKSTREMOW
for i in (sk):
    cv2.circle(im2,(xaxis[i],yaxis[i]), 3, (0,0,255), -1)
#print [pts]
# Draw a diagonal blue line with thickness of 5 px
#cv2.line(img,(0,0),(511,511),(255,0,255),5)
#cont, hi = cv2.findContours(img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE, offset=(0, 0))
#cv2.drawContours(img, cont,-1,(255,0,0))# hole_color=(0,0,255), max_level=1, thickness=1, lineType=8, offset=(0, 0))

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',im2)
cv2.waitKey(0)
cv2.destroyAllWindows()
#"""

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        bg=frame
        break
t=10
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    #im3 = cv2.compare(frame,bg, cmpop=cv2.CMP_EQ)
    im3 = cv2.absdiff(frame,bg)
    gray = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

    #sob=cv2.Sobel(frame, ddepth = -1,dx = 0,dy = 1)
    ret,thresh = cv2.threshold(gray,t,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    maxi=0
    i=1
    if len(contours)>0:
        while i!=len(contours):
            if len(contours[i])>len(contours[maxi]):
                maxi=i
            i=i+1
    #print len(contours)
        cv2.drawContours(frame,[contours[maxi]],-1,(0,255,0),3)
        #------------------------------------------------------ KONWERSJA DANYCH Z FIND CONTOURS DO OSI X / Y
        xaxis=[]
        yaxis=[]
        for i in range(len(contours[maxi])):
            xaxis.append( (((contours[maxi])[i])[0])[0])
            yaxis.append( (((contours[maxi])[i])[0])[1])
        #------------------------------------------------------ CENTROID
        x=(max(xaxis) + min(xaxis))/2
        y=(max(yaxis) + min(yaxis))/2
        cv2.circle(frame,(x,y), 5, (0,0,255), -1)
        #------------------------------------------------------ ODLEGLOSCI WSZYSTKICH PUNKTOW OD CENTROIDA
        odl=[]
        for i in range(len(xaxis)):
            odl.append(  np.sqrt((x-xaxis[i])**2 + (y-yaxis[i])**2  )   )
        #------------------------------------------------------ DETEKCJA EKSTREMOW
        #odl1 = np.array(odl)
        #sss=argrelmax(odl1)
        maxtab, mintab = peakdet(odl,30.0)
        if len(maxtab)>0:
            #------------------------------------------------------ WYCIAGANIE POTRZEBNYCH DANYCH Z WYNIKU PEAKDET
            sk=[]
            for i in (maxtab[:,0]):
                sk.append(int(i))
            #------------------------------------------------------ RYSOWANIE EKSTREMOW
            for i in (sk):
                cv2.circle(frame,(xaxis[i],yaxis[i]), 5, (0,0,255), -1)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('a'):
        t=t+1
        print t
    elif cv2.waitKey(1) & 0xFF == ord('z'):
        t=t-1
        print t

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()#"""