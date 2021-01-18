import cv2
import numpy as np

def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h,w) = src.shape
    h_dst = int(h*scale)
    w_dst = int(w*scale)
    dst = np.zeros((h_dst, w_dst), np.uint8)

    for row in range(h_dst):
        x = row / scale
        x2 = int(np.ceil(x))
        x1 = int(np.floor(x))
        if (x2 > h - 1):
            x2 = h - 1
        for col in range(w_dst):
            y = col / scale
            y2 = int(np.ceil(y))
            y1 = int(np.floor(y))
            if(y2>w-1):
                y2=w-1
            if((x2==x1)or (y1==y2)):
                if(x1==x2 and y1==y2) :
                    dst[row, col] = src[x1, y1]
                elif(y2==y1):
                    sum = src[x1, y1] * (x2 - x) + src[x2, y1] * (x - x1)
                    sum = max(min(sum, 255), 0)
                    dst[row, col] = sum
                elif(x1==x2 ):
                    sum = src[x1, y2] * (y - y1) + src[x1, y1] * (y2 - y)
                    sum = max(min(sum, 255), 0)
                    dst[row, col] = sum
            else:
                sum = src[x1,y1]*(y2-y)*(x2-x)+src[x1,y2]*(y-y1)*(x2-x)+src[x2,y1]*(y2-y)*(x-x1)+src[x2,y2]*(y-y1)*(x-x1)
                sum = max(min(sum,255),0)
                dst[row, col] = sum

    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    #이미지 크기 ??x??로 변경
    my_dst_mini = my_bilinear(src, 0.25)

    #이미지 크기 512x512로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, 4)


    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()
