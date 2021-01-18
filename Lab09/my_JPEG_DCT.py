import cv2
import numpy as np
import my_padding as my_p

def my_normalize(src):
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)

def my_DCT(src, n=8):
    (h, w) = src.shape
    dct_img = (src.copy()).astype(np.float)
    block_h, block_w = int(np.ceil(h / n)), int(np.ceil(w / n))

    pad_img = my_p.my_padding(dct_img, (np.abs(block_h * n - h),np.abs(block_w * n - w)), 'repetition')
    (pad_row, pad_col) = pad_img.shape
    pad = np.zeros((pad_row, pad_col))
    dst = np.zeros((h,w))
    block_after = np.zeros((n,n))

    block = np.zeros((n, n))
    for b_row in range(pad_row // n):
        for b_col in range(pad_col // n):
            for i, row in enumerate(range(int(b_row * n), (b_row + 1) * n)):
                for j, col in enumerate(range(int(b_col * n), (b_col + 1) * n)):
                    block[i, j] = pad_img[row, col]
            for u in range(n):
                for v in range(n):
                    if u == 0 and v == 0:
                        cuv = np.sqrt(1 / n) * np.sqrt(1 / n)
                    elif u == 0 or v == 0:
                        cuv = np.sqrt(1 / n) * np.sqrt(2 / n)
                    else:
                        cuv = np.sqrt(2 / n) * np.sqrt(2 / n)
                    sum = 0
                    for x in range(n):
                        for y in range(n):
                            cos = np.cos(((2 * y + 1) * v * np.pi) / (2 * n)) * np.cos(((2 * x + 1) * u * np.pi) / (2 * n))
                            tmp = block[x,y]*cos
                            sum = sum+ tmp
                    block_after[u,v] = cuv*sum
            pad[b_row*n:(b_row+1)*n, b_col*n:(b_col+1)*n] = block_after
    dst[:,:] = pad[:h,:w]

    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_DCT(src, 8)

    dst = my_normalize(dst)

    cv2.imshow('original',src)
    cv2.imshow('my DCT', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


