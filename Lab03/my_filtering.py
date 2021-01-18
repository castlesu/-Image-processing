import cv2
import numpy as np

def my_filtering(src, ftype, fsize):
    (h, w) = src.shape
    dst = np.zeros((h, w))
    row , col = fsize
    total = row*col
    mask = [[] for i in range(col)]

    if ftype == 'average':
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                       #
        ###################################################
        for i in range(col):
            for j in range(row):
                mask[i].append(1/total)
        mask = np.array(mask)
        #mask 확인
        # print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                      #
        ##################################################
        for i in range(col):
            for j in range(row):
                if(i==int(col/2) and j==int(row/2)):
                    mask[i].append(2-(1/total))
                else:
                    mask[i].append(0-(1/total))
        mask = np.array(mask)
        #mask 확인
        # print(mask)

    #########################################################
    # TODO                                                  #
    # dst 완성                                              #
    # dst : filtering 결과 image                            #
    #########################################################
    #

    for c in range(h):
        for r in range(w):
            for i in range(col):
                for j in range(row):
                    if((c+i-int(col/2))<0 ):
                        dst[c, r] += 0*mask[i,j]
                    elif((r+j-int(row/2)) <0):
                        dst[c, r] += 0 * mask[i, j]
                    elif ((r+j-int(row/2))>(w-1)):
                        dst[c, r] += 0 * mask[i, j]
                    elif((c+i-int(col/2)) >(h-1)):
                        dst[c, r] += 0*mask[i,j]
                    else:
                         dst[c,r]+=src[c+i-int(col/2),r+j-int(row/2)]*mask[i,j]
    sum =0
    for row in range(h):
        for col in range(w):
            sum += src[row,col] + mask[row,col]

    for row in range(h):
        for col in range(w):
            if (dst[row,col] <0):
                dst[row,col] == 0
            elif(dst[row,col] > 255):
                dst[row,col] == 255

    # print(dst)
    dst = (dst+0.5).astype(np.uint8)
    # dst2 = cv2.filter2D(src,-1,mask)
    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # # 3x3 filter
    dst_average1 = my_filtering(src, 'average', (3,3))
    dst_sharpening1 = my_filtering(src, 'sharpening', (3,3))

    # #
    # # #원하는 크기로 설정
    dst_average2 = my_filtering(src, 'average', (5,5))
    dst_sharpening2 = my_filtering(src, 'sharpening', (5,5))

    # #
    # # # 11x13 filter
    dst_average3 = my_filtering(src, 'average', (11,13))
    dst_sharpening3 = my_filtering(src, 'sharpening', (11,13))
    # #
    cv2.imshow('original', src)
    cv2.imshow('average filter 3x3', dst_average1)
    cv2.imshow('sharpening filter 3x3', dst_sharpening1)
    cv2.imshow('average filter 5x5', dst_average2)
    cv2.imshow('sharpening filter 5x5', dst_sharpening2)
    cv2.imshow('average filter 11x13', dst_average3)
    cv2.imshow('sharpening filter 11x13', dst_sharpening3)




    cv2.waitKey()
    cv2.destroyAllWindows()
