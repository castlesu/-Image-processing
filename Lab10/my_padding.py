import numpy as np
import cv2

def my_padding(src, pad_shape, pad_type = 'repetition'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h +  p_h, w + p_w))
    pad_img[: h, :w] = src
    pad_img[h:, : w] = src[h - 1, :]
    pad_img[:, w:] = pad_img[:, w - 1:w]

    return pad_img

if __name__=='__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    #zero padding
    # my_pad_img = my_padding(src, (20, 20))

    #repetition padding
    my_pad_img = my_padding(src, (3, 3), 'repetition')

    #데이터타입 uint8로 변경
    my_pad_img = (my_pad_img + 0.5).astype(np.uint8)
    cv2.imshow('original', src)
    cv2.imshow('my padding img', my_pad_img)

    cv2.waitKey()
    cv2.destroyAllWindows()


