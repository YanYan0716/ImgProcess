"""
reference:https://www.youtube.com/watch?v=wyb5S8QsCSA&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=19
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# 亮度量化表 固定值
# 还有色差变化表 reference：https://blog.csdn.net/w394221268/article/details/52232933
INTENSITY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])


def dct_block(block, size=8, ratio=0.5):
    """
    离散余弦变换的一个block
    :param block: 8*8的矩阵
    :param size: 一般为8 不会变化
    :param ratio:压缩比例
    :return:
    """
    g_block = np.zeros(block.shape, dtype=np.float)
    for u in range(int(size*ratio)):
        for v in range(int(size*ratio)):
            if u == 0:
                a_u = 1 / np.sqrt(2)
            else:
                a_u = 1

            if v == 0:
                a_v = 1 / np.sqrt(2)
            else:
                a_v = 1
            a = 0
            for x in range(size):
                for y in range(size):
                    a += block[x, y] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
            g_block[u, v] = a_u * a_v * a / 4

    return g_block


def DCT(img, size=8, ratio=0.5):
    '''
    离散余弦变换 https://www.freesion.com/article/3320573502/
    :param img:
    :return:
    '''
    gray_img = img
    [H, W] = gray_img.shape
    normal_img = np.array(gray_img, dtype=np.int) - 128
    new_img = np.ones(normal_img.shape, dtype=np.float)

    # 二维离散余弦变换
    for h in range(0, H, 8):
        for w in range(0, W, 8):
            block = normal_img[h:h+8, w:w+8]
            new_img[h:h+8, w:w+8] = dct_block(block, size, ratio)
    return new_img


def idct_block(block, size=8):
    """
    离散反余弦变换的一个block
    :param block: 8*8的img 一般不会有变化
    :param size:
    :return:
    """
    block = np.round(block*INTENSITY).astype(np.float)
    g_block = np.zeros(block.shape, dtype=np.float)
    for x in range(size):
        for y in range(size):
            a = 0
            for u in range(size):
                for v in range(size):
                    if u == 0:
                        a_u = 1/np.sqrt(2)
                    else:
                        a_u = 1

                    if v == 0:
                        a_v = 1 / np.sqrt(2)
                    else:
                        a_v = 1

                    a += a_u*a_v*block[u, v]*np.cos((2*x+1)*u*np.pi/16)*np.cos((2*y+1)*v*np.pi/16)
            g_block[x, y] = (a/4)
    return g_block


def IDCT(img, size=8):
    '''
    离散反余弦变换
    :param img:
    :return:
    '''
    [H, W] = img.shape
    new_img = np.ones(img.shape, dtype=np.float)

    # 二维离散反余弦变换
    for h in range(0, H, 8):
        for w in range(0, W, 8):
            block = img[h:h+8, w:w+8]
            new_img[h:h+8, w:w+8] = idct_block(block, size)
    return new_img+128


figure = plt.figure(num='LossyImageCompression', figsize=(6, 6))
img = cv2.imread('./data/IMG20.jpg', 0)
plt.subplot(2, 2, 1)
plt.title('origin')
plt.imshow(img, cmap='gray')

# opencv的离散余弦变换
img_dct = cv2.dct(img.astype('float'))
img_dct_log = np.log(abs(img_dct))
plt.subplot(2, 2, 2)
plt.title('dct transform')
plt.imshow(img_dct_log, cmap='gray')

recor_temp = img_dct[0:100, 0:100]
recor_temp2 = np.zeros(img.shape)
recor_temp2[0:100, 0:100] = recor_temp
img_recor = cv2.idct(recor_temp2)
plt.subplot(2, 2, 3)
plt.title('recons')
plt.imshow(img_recor, cmap='gray')

P = DCT(img, size=8)
Q = IDCT(P, size=8)
plt.subplot(2, 2, 4)
plt.title('recons_self')
plt.imshow(Q, cmap='gray')

plt.show()

# '''===========离散余弦变换的小例子=================='''
# def dct_block(block):
#     g_block = np.zeros(block.shape, dtype=np.float)
#     for u in range(8):
#         for v in range(8):
#             if u == 0:
#                 a_u = 1/np.sqrt(2)
#             else:
#                 a_u = 1
#
#             if v == 0:
#                 a_v = 1 / np.sqrt(2)
#             else:
#                 a_v = 1
#             a = 0
#             for x in range(8):
#                 for y in range(8):
#                     a += block[x, y]*np.cos((2*x+1)*u*np.pi/16)*np.cos((2*y+1)*v*np.pi/16)
#             g_block[u, v] = a_u*a_v*a/4
#     return g_block
#
#
# block = np.array([[52, 55, 61, 66, 70, 61, 64, 73],
#                   [63, 59, 55, 90, 109, 85, 69, 72],
#                   [62, 59, 68, 113, 144, 104, 66, 73],
#                   [63, 58, 71, 122, 154, 106, 70, 69],
#                   [67, 61, 68, 104, 126, 88, 68, 70],
#                   [79, 65, 60, 70, 77, 68, 58, 75],
#                   [85, 71, 64, 59, 55, 61, 65, 83],
#                   [87, 79, 69, 68, 65, 76, 78, 94]])
#
# b_block = np.array(block-128, dtype=np.float)
# g = dct_block(b_block)
# Q = np.array(np.round(g/INTENSITY), dtype=np.int)
#
#
# def idct_block(block):
#     block = np.round(block*INTENSITY).astype(np.float)
#     g_block = np.zeros(block.shape, dtype=np.float)
#     for x in range(8):
#         for y in range(8):
#             a = 0
#             for u in range(8):
#                 for v in range(8):
#                     if u == 0:
#                         a_u = 1/np.sqrt(2)
#                     else:
#                         a_u = 1
#
#                     if v == 0:
#                         a_v = 1 / np.sqrt(2)
#                     else:
#                         a_v = 1
#
#                     a += a_u*a_v*block[u, v]*np.cos((2*x+1)*u*np.pi/16)*np.cos((2*y+1)*v*np.pi/16)
#             g_block[x, y] = (a/4)
#     return g_block
#
#
# m = np.round(idct_block(Q))
# print(m)