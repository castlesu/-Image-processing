import numpy as np
import cv2
import my_padding as my_p

def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance

def my_DCT(src, n=8):
    (h, w) = src.shape
    dct_img = (src.copy()).astype(np.float)
    block_h, block_w = int(np.ceil(h / n)), int(np.ceil(w / n))

    pad_img = my_p.my_padding(dct_img, (np.abs((block_h * n) - h),np.abs((block_w * n) - w)), 'repetition')
    (pad_row, pad_col) = pad_img.shape
    dst = np.zeros((pad_row, pad_col))
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
            dst[b_row*n:(b_row+1)*n, b_col*n:(b_col+1)*n] = block_after

    return dst

def my_JPEG_encoding(src, block_size=8):
    #####################################################
    # TODO                                              #
    # my_block_encoding 완성                            #
    # 입력변수는 알아서 설정(단, block_size는 8로 설정)   #
    # return                                            #
    # zigzag_value : encoding 결과(zigzag까지)          #
    #####################################################
    src = src - 128
    dct = my_DCT(src, block_size)
    (h,w) = dct.shape
    luminance= Quantization_Luminance()

    zigzag_value = []
    # block_after = np.zeros((block_size, block_size))
    divide = np.zeros((h,w))
    block = np.zeros((block_size, block_size))
    # a = []
    for b_row in range(h // block_size):
        for b_col in range(w // block_size):
            for i, row in enumerate(range(int(b_row * block_size), (b_row + 1) * block_size)):
                for j, col in enumerate(range(int(b_col * block_size), (b_col + 1) * block_size)):
                    block[i, j] = dct[row, col]
            block_after = np.round(block/Quantization_Luminance().astype(np.float))

            divide[b_row * block_size:(b_row + 1) * block_size, b_col * block_size:(b_col + 1) * block_size] = block_after
            zigzag_tmp = (np.concatenate([np.diagonal(block_after[::-1,:], i)[::(-(2*(i % 2)-1))] for i in range(1-block_size, block_size)]))
            tmp = []
            for i in range(block_size*block_size):
                if i ==0:
                    if zigzag_tmp[i] != 0:
                        tmp.append(zigzag_tmp[i])
                    elif zigzag_tmp[i] == 0:
                        j = i
                        while (1):
                            if zigzag_tmp[j + 1] == 0:
                                j = j + 1
                            else:
                                break
                            if j == ((block_size * block_size) - 1):
                                tmp.append(np.nan)
                                break
                if np.isnan(tmp).any() == False :
                    if i != 0:
                        tmp.append(zigzag_tmp[i])
                    j = i
                    while(1):
                        if zigzag_tmp[j+1] == 0:
                            j = j+1
                        else:
                            break
                        if j == ((block_size * block_size)-1):
                            tmp.append(np.nan)
                            break
            zigzag_value.append(tmp)
    return zigzag_value , divide

def zigzag(zigzag,block_size):
    n = block_size
    block = np.zeros((n,n))
    i, j = 0, 0
    num = 0
    while j != (n - 1):
        block[i,j] = zigzag[num]
        num = num+1

        if i == 0 and (j & 1):
            j += 1
            continue
        if j == 0 and (i & 1) == 0:
            i += 1
            continue
        if (i ^ j) & 1:
            i -= 1
            j += 1
            continue
        if (i ^ j) & 1 == 0:
            i += 1
            j -= 1
            continue
    while i != (n - 1) or j != (n - 1):

        block[i, j] = zigzag[num]
        num = num+ 1
        if i == (n - 1) and (j & 1):
            j += 1
            continue
        if j == (n - 1) and (i & 1) == 0:
            i += 1
            continue
        if (i ^ j) & 1:
            i -= 1
            j += 1
            continue
        if (i ^ j) & 1 == 0:
            i += 1
            j -= 1
            continue

    return block
def my_IDCT(src,n=8):
    (h, w) = src.shape
    # idct_img = (src.copy()).astype(np.float)

    dst = np.zeros((h, w))
    block_after = np.zeros((n,n))
    for b_row in range(h// n):
        for b_col in range(w // n):
            block = src[b_row*n:(b_row+1)*n,b_col*n:(b_col+1)*n]
            for x in range(n):
                for y in range(n):
                    sum = 0
                    for u in range(n):
                        for v in range(n):
                            if u == 0 and v == 0:
                                cuv = np.sqrt(1 / n) * np.sqrt(1 / n)
                            elif u == 0 or v == 0:
                                cuv = np.sqrt(1 / n) * np.sqrt(2 / n)
                            else:
                                cuv = np.sqrt(2 / n) * np.sqrt(2 / n)
                            cos = np.cos(((2 * y + 1) * v * np.pi) / (2*n)) * np.cos(((2 * x + 1) * u * np.pi) / (2*n))
                            tmp = block[u,v]*cos*cuv
                            sum = sum+ tmp
                    block_after[x,y] = sum
            dst[b_row*n:(b_row+1)*n, b_col*n:(b_col+1)*n] = block_after
    return dst

def my_JPEG_decoding(divide, zigzag_value, block_size=8):
    #####################################################
    # TODO                                              #
    # my_JPEG_decoding 완성                             #
    # 입력변수는 알아서 설정(단, block_size는 8로 설정)   #
    # return                                            #
    # dst : decoding 결과 이미지                         #
    #####################################################
    # (h,w) = src.shape
    # dst = np.zeros((h,w))
    zigzag_decode = np.zeros((int(np.sqrt(len(zigzag_value)))*block_size,int(np.sqrt(len(zigzag_value)))*block_size))
    b_row,b_col = zigzag_decode.shape[0]//block_size, zigzag_decode.shape[0]//block_size
    # test = np.zeros(
    #     (int(np.sqrt(len(zigzag_value))) * block_size, int(np.sqrt(len(zigzag_value))) * block_size))

    zigzag_value = zigzag_value
    zigzag_block = []
    for i in range(len(zigzag_value)):
        tmp = []
        for b in range(block_size*block_size):
            if len(tmp) == block_size*block_size:
                break
            elif len(tmp) != block_size*block_size :
                if np.isnan(zigzag_value[i][b]) == False:
                    tmp.append(zigzag_value[i][b])
                elif np.isnan(zigzag_value[i][b]) == True:
                    j = b
                    while(1):
                        tmp.append(0)
                        j = j+1
                        if j == (block_size * block_size) :
                            break
        block = zigzag(tmp,block_size)
        zigzag_block.append(block)
    n = 0
    for row in range(b_row):
        for col in range(b_col):
            block = zigzag_block[n]
            block_after = block*Quantization_Luminance()
            zigzag_decode[row*block_size:(row+1)*block_size,col*block_size:(col+1)*block_size] = block_after
            n= n+1

    zigzag_decode = my_IDCT(zigzag_decode,block_size)
    zigzag_decode = zigzag_decode+128

    return zigzag_decode.astype(np.uint8)
def my_normalize(src):
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)

if __name__ == '__main__':

    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    n=8
    src = src.astype(np.float)
    # zigzag_value , devide = my_JPEG_encoding(src,n)
    # print(zigzag_value[:10])
    #
    # dst = my_JPEG_decoding(devide, zigzag_value,n)
    # dst = my_normalize(dst)

    src = src.astype(np.uint8)
    # dst = dst.astype(np.uint8)

    cv2.imshow('original', src)
    # cv2.imshow('result', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


