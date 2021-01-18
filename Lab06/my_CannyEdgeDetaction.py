import cv2
import numpy as np
import my_padding as my_p

#low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1, pad_type='zero'): #DOG적용
    #########################################################################################
    # TODO                                                                                   #
    # apply_lowNhigh_pass_filter 완성                                                        #
    # Ix : image에 DoG_x filter 적용 or gaussian filter 적용된 이미지에 sobel_x filter 적용    #
    # Iy : image에 DoG_y filter 적용 or gaussian filter 적용된 이미지에 sobel_y filter 적용    #
    ###########################################################################################
    #low-pass filter를 이용하여 blur효과
    #high-pass filter를 이용하여 edge 검출
    #gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨
    (h,w) = src.shape

    y, x = np.mgrid[-(fsize // 2):(fsize // 2) + 1, -(fsize // 2):(fsize // 2) + 1]
    DoGx = (-x / sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    DoGy = (-y / sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    (dh,dw) = DoGx.shape
    pad_img = my_p.my_padding(src, (dh // 2, dw // 2), pad_type)

    Ix = np.zeros((h,w))
    Iy = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            Ix[row,col] = np.sum(pad_img[row:row + dh, col:col + dw] * DoGx)
            Iy[row,col] = np.sum(pad_img[row:row + dh, col:col + dw] * DoGy)

    return Ix, Iy

#Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ##################################################
    # TODO                                           #
    # calcMagnitude 완성                             #
    # magnitude : ix와 iy의 magnitude를 계산         #
    #################################################
    # magnitude = np.hypot(Ix,Iy)
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    return magnitude

#Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    #######################################
    # TODO                                #
    # calcAngle 완성                      #
    # angle     : ix와 iy의 angle         #
    #######################################

    angle = np.arctan2(Iy,Ix)
    angle = angle+np.pi
    return angle

#non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                      #
    # larger_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)         #
    ####################################################################################
    (h, w) = magnitude.shape
    larger_magnitude = np.zeros((h, w))

    for row in range(1, h - 1):
        for col in range(1, w - 1):
            theta = angle[row, col]
            if (0 <= theta < np.pi / 4) or (np.pi <= theta < 5 / 4 * np.pi):
                t = np.tan(theta)
                before = magnitude[row + 1, col + 1] * t + magnitude[row, col + 1] * (1 - t)
                after = magnitude[row - 1, col - 1] * t + magnitude[row, col - 1] * (1 - t)
            elif (np.pi / 4 <= theta < np.pi / 2) or (5 / 4 * np.pi <= theta < 3 / 2 * np.pi):
                t = 1 / np.tan(theta)
                before = magnitude[row + 1, col + 1] * t + magnitude[row + 1, col] * (1 - t)
                after = magnitude[row - 1, col - 1] * t + magnitude[row - 1, col] * (1 - t)
            elif (np.pi / 2 <= theta < 3 / 4 * np.pi) or (3 / 2 * np.pi <= theta < 7 / 4 * np.pi):
                t = -1 / np.tan(theta)
                before = magnitude[row + 1, col - 1] * t + magnitude[row + 1, col] * (1 - t)
                after = magnitude[row - 1, col + 1] * t + magnitude[row - 1, col] * (1 - t)
            elif (3 / 4 * np.pi <= theta < np.pi) or (7 / 4 * np.pi <= theta < np.pi * 2):
                t = -np.tan(theta)
                before = magnitude[row + 1, col - 1] * t + magnitude[row, col - 1] * (1 - t)
                after = magnitude[row - 1, col + 1] * t + magnitude[row, col + 1] * (1 - t)

            max_value = max([before, after, magnitude[row, col]])
            if max_value == magnitude[row, col]:
                larger_magnitude[row, col] = magnitude[row,col]
            else:
                larger_magnitude[row, col] = 0
    #larger_magnitude값을 0~255의 uint8로 변환
    larger_magnitude = (larger_magnitude/np.max(larger_magnitude)*255).astype(np.uint8)
    cv2.imshow('num',larger_magnitude)

    return larger_magnitude



#double_thresholding 수행 high threshold value는 내장함수(otsu방식 이용)를 사용하여 구하고 low threshold값은 (high threshold * 0.4)로 구한다
def double_thresholding(src):
    (h,w) = src.shape
    dst = np.zeros((h,w))
    high_threshold_value,_ = cv2.threshold(src,0,255,cv2.THRESH_OTSU)
    low_threshold_value = high_threshold_value*0.4

    strong_row,strong_col = np.where(src>=high_threshold_value)
    non_row,non_col = np.where(src<=low_threshold_value)
    weak_row,weak_col = np.where((src<high_threshold_value)&(src>low_threshold_value))
    dst[strong_row,strong_col] = 255 #strong value = 255
    dst[non_row,non_col] = 0
    dst[weak_row,weak_col] = 128 #weak value = 128
    change=True
    while (change):
        change = False
        for row in range(1, h-1):
            for col in range(1, w-1):
                if(dst[row,col] == 128):
                    if ((dst[row + 1, col] == 255) or (dst[row, col + 1] == 255)
                            or (dst[row - 1, col] == 255) or (dst[row, col - 1] == 255)
                            or (dst[row + 1, col + 1] == 255) or (dst[row - 1, col - 1] == 255)
                            or (dst[row - 1, col + 1] == 255) or (dst[row + 1, col - 1] == 255)):
                        dst[row,col] =255
                if (dst[row, col] == 255):
                    if ((dst[row + 1, col] == 128) or (dst[row, col + 1] == 128)
                            or (dst[row - 1, col] == 128) or (dst[row, col - 1] == 128)
                            or (dst[row + 1, col + 1] == 128) or (dst[row - 1, col - 1] == 128)
                            or (dst[row - 1, col + 1] == 128) or (dst[row + 1, col - 1] == 128)):
                        change = True
    weekrow,weekcol = np.where(dst==128)
    dst[weekrow,weekcol] = 0
    return dst

def my_canny_edge_detection(src, fsize=5, sigma=1, pad_type='zero'):
    #low-pass filter를 이용하여 blur효과
    #high-pass filter를 이용하여 edge 검출
    #gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma, pad_type)

    #magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    #non-maximum suppression 수행
    larger_magnitude = non_maximum_supression(magnitude, angle)

    #진짜 edge만 남김
    dst = double_thresholding(larger_magnitude)

    return dst


if __name__ =='__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src)
    # test = cv2.imread('double_threshold_test_img.png',cv2.IMREAD_GRAYSCALE)
    # dst2 = double_thresholding(test)
    #
    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
