import cv2
import numpy as np

def my_bgr2gray(src):
    '''
    :param src:컬러 이미지
    :return dst1, dst2, dst3:흑백 이미지
    '''
    img1 = src
    img2 = src
    img3 = src

    #cvtColor() 함수 이용
    dst1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    #########################
    # TODO                  #
    # dst2, dst3 채우기     #
    #########################

    #dst2는 B, G, R채널 각각 1/3씩 사용
    (h,v,c) = img2.shape
    dst2 = np.zeros((h,v))
    dst2 = (img2[:,:,0]*0.114+img2[:,:,1]*0.587+img2[:,:,2]*0.299)


    #dst3은 B, G, R채널 중 하나의 채널만 사용(B,G,R중 원하는거 아무거나)
    b,g,r = cv2.split(img3)
    dst3=r

    #dst2 반올림 np.round를 사용해도 무관
    dst2 = np.round(dst2).astype(np.uint8)
    return dst1, dst2, dst3


#아래의 이미지 3개 다 해보기
fruits = cv2.imread('fruits.jpg')
lena = cv2.imread('Lena.png')
penguins = cv2.imread('Penguins.png')

dst1, dst2, dst3 = my_bgr2gray(fruits)

cv2.imshow('original-F', fruits)
cv2.imshow('gray(cvtColor)-F', dst1)
cv2.imshow('gray(1/3)-F', dst2)
cv2.imshow('gray(one channel)-F', dst3)

dst11, dst22, dst33 = my_bgr2gray(lena)

cv2.imshow('original-L', lena)
cv2.imshow('gray(cvtColor)-L', dst11)
cv2.imshow('gray(1/3)-L', dst22)
cv2.imshow('gray(one channel)-L', dst33)

dst111, dst222, dst333 = my_bgr2gray(penguins)

cv2.imshow('original-P', penguins)
cv2.imshow('gray(cvtColor)-P', dst111)
cv2.imshow('gray(1/3)-P', dst222)
cv2.imshow('gray(one channel)-P', dst333)

cv2.waitKey()
cv2.destroyAllWindows()