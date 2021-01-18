import cv2
import numpy as np
import my_padding as my_p

def dilation(B, S):
    ###############################################
    # TODO                                        #
    # dilation 함수 완성                          #
    ###############################################
    (h, w) = B.shape
    (s_h, s_w) = S.shape
    img_dilation = np.zeros((h, w))
    pad_img = my_p.my_padding(B, (s_h // 2, s_w // 2), 'zero')

    for row in range(h):
        for col in range(w):
            compare = []
            compare.append(pad_img[row:row+s_h,col:col+s_w]==S)
            compare = np.array(compare).flatten()
            chk = compare.any()
            if (chk==True):
                img_dilation[row,col] = 255
            else:
                img_dilation[row,col] = 0


    return img_dilation

def erosion(B, S):
    ###############################################
    # TODO                                        #
    # erosion 함수 완성                           #
    ###############################################

    (h, w) = B.shape
    (s_h, s_w) = S.shape
    img_erosion = np.zeros((h, w))
    pad_img = my_p.my_padding(B, (s_h // 2, s_w // 2), 'zero')

    for row in range(h):
        for col in range(w):
            compare = []
            compare.append(pad_img[row:row+s_h,col:col+s_w]==S)
            compare = np.array(compare).flatten()
            chk = compare.all()
            if (chk==True):
                img_erosion[row,col] = 255
            else:
                img_erosion[row,col] = 0

    return img_erosion



def opening(B, S):
    ###############################################
    # TODO                                        #
    # opening 함수 완성                           #
    ###############################################
    E = erosion(B, S)
    img_opening = dilation(E, S)

    return img_opening

def closing(B, S):
    ###############################################
    # TODO                                        #
    # closing 함수 완성                           #
    ###############################################
    D = dilation(B, S)
    img_closing = erosion(D, S)
    return img_closing


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [255, 255, 255, 255, 255, 255, 255, 0],
         [0, 0, 0, 255, 255, 255, 255, 0],
         [0, 0, 0, 255, 255, 255, 255, 0],
         [0, 0, 255, 255, 255, 255, 255, 0],
         [0, 0, 0, 255, 255, 255, 255, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]
    , dtype = np.uint8)

    S = np.array(
        [[255, 255, 255],
         [255, 255, 255],
         [255, 255, 255]]
    , dtype = np.uint8)

    cv2.imwrite('morphology_B.png', B)

    img_dilation = dilation(B, S)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(B, S)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(B, S)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)


