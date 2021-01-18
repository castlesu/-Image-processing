import numpy as np
import cv2
import my_padding as my_p

def my_get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    #2차 gaussian mask 생성
    gaus2D =   1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))
    #mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)
    print(gaus2D)
    return gaus2D

def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    (m_h, m_w) = mask.shape
    pad_img = my_p.my_padding(src, (m_h // 2, m_w // 2), pad_type)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)
    return dst

def my_normalize(src):
    dst = src.copy()
    dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)

def my_bilateral(src, msize, sigma, sigma_r, pad_type='zero'):
    ############################################
    # TODO                                     #
    # my_bilateral 함수 완성                   #
    # src : 원본 image                         #
    # msize : mask size                        #
    # sigma : sigma_x, sigma_y 값              #
    # sigma_r : sigma_r값                      #
    # pad_type : padding type                  #
    # dst : bilateral filtering 결과 image     #
    ############################################
    (h,w) = src.shape
    mask = np.zeros((msize,msize))
    pad_img = my_p.my_padding(src, (msize // 2, msize // 2), pad_type)
    dst = np.zeros((h,w))
    divmask = int(msize//2)

    for row in range(h):
        for col in range(w):
            for m_h in range(msize):
                for m_w in range(msize):
                    mask[m_h,m_w] = np.exp(-(((row + divmask - (m_h + row)) ** 2 + (col + divmask - (m_w+col)) ** 2) / (2 * sigma ** 2) + (
                            src[row, col] - pad_img[row+m_h, col+m_w]) ** 2 / (2 * sigma_r ** 2)))
            mask /= np.sum(mask)
            dst[row, col] = np.sum(pad_img[row:row+ msize, col:col + msize] * mask)
    dst = (dst + 0.5).astype(np.uint8)

    return dst



if __name__ == '__main__':
    src = cv2.imread('Penguins_noise.png', cv2.IMREAD_GRAYSCALE)
    dst = my_bilateral(src, 5, 20, 30)

    gaus2D = my_get_Gaussian2D_mask(5 , sigma = 1)
    dst_gaus2D= my_filtering(src, gaus2D)
    dst_gaus2D = my_normalize(dst_gaus2D)

    cv2.imshow('original', src)
    cv2.imshow('my gaussian', dst_gaus2D)
    cv2.imshow('my bilateral', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()

