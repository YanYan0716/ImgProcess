'''
https://www.youtube.com/watch?v=IcBzsP-fvPo&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=15&t=3239s
'''


import numpy as np
import matplotlib.pyplot as plt
import cv2


def binaryImg(img, threshold):
    '''
    图像二值化
    :param img: 图片的格式 [H, W, C]
    :param threshold: 二值化的阈值(0~255)
    :return: 二值化的图像
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    [H, W] = gray_img.shape
    binary_img = np.ones(shape=gray_img.shape)
    for i in range(H):
        for j in range(W):
            if gray_img[i, j] >= threshold:
                binary_img[i, j] = 0
    return binary_img


def earnKernel(ksize):
    assert ksize % 2 == 1, 'ksize should be odd '
    a = np.ones(shape=(int(ksize), int(ksize)))
    diag1 = np.triu(a, k=-int((ksize - 1) / 2))
    diag2 = np.tril(a, k=int((ksize - 1) / 2))
    diag = diag1 * diag2
    diag_ = cv2.rotate(diag, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    return diag * diag_


def erosion(img, ksize):
    kernel = earnKernel(ksize)
    (H, W) = img.shape

    kernel_sum = np.sum(kernel)
    new_img = np.zeros(shape=img.shape)
    a = int((ksize - 1) / 2)
    for i in range(0 + a + 1, H - a):
        for j in range(0 + a + 1, W - a):
            if np.sum(img[i - a - 1:i + a, j - a - 1:j + a] * kernel) == kernel_sum:
                new_img[i, j] = img[i, j]

    return new_img


def dilate(img, ksize):
    kernel = earnKernel(ksize)
    (H, W) = img.shape

    new_img = np.zeros(shape=img.shape)
    a = int((ksize - 1) / 2)
    for i in range(0 + a + 1, H - a):
        for j in range(0 + a + 1, W - a):
            if np.sum(img[i - a - 1:i + a, j - a - 1:j + a] * kernel) > 0:
                new_img[i, j] = 1
    return new_img


img = cv2.imread('./data/IMG17.jpg')


plt.figure(num='image erosion & dilate', figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.title('origin image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

bImg = binaryImg(img, threshold=200)
plt.subplot(2, 2, 2)
plt.title('binary image')
plt.imshow(bImg, cmap='gray')

erosion_img = erosion(bImg, 5)
plt.subplot(2, 2, 3)
plt.title('erosion image')
plt.imshow(erosion_img, cmap='gray')

dilate_img = dilate(bImg, 5)
plt.subplot(2, 2, 4)
plt.title('dilate image')
plt.imshow(dilate_img, cmap='gray')
plt.show()


def opening(img, ksize):
    '''
    图像的开运算
    :param img:
    :param ksize:
    :return:
    '''
    img = erosion(img, ksize)
    return dilate(img, ksize)


def closing(img, ksize):
    '''
    图像的闭运算
    :param img:
    :param ksize:
    :return:
    '''
    img = dilate(img, ksize)
    return erosion(img, ksize)


plt.figure(num='image opening & closing', figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.title('origin image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

bImg = binaryImg(img, threshold=200)
plt.subplot(2, 2, 2)
plt.title('binary image')
plt.imshow(bImg, cmap='gray')

oimg = opening(bImg, 5)
plt.subplot(2, 2, 3)
plt.title('opening')
plt.imshow(oimg, cmap='gray')

cimg = closing(bImg, 5)
plt.subplot(2, 2, 4)
plt.title('closing')
plt.imshow(cimg, cmap='gray')
plt.show()


def edgedetect(img, ksize):
    '''
    图像的边缘提取
    :param img:
    :param ksize:
    :return:
    '''
    img_ = erosion(img, ksize)
    # 异或运算 input数据类型为bool
    res = np.bitwise_xor(np.bool8(img), np.bool8(img_))
    return np.array(res, dtype=np.uint8)


plt.figure(num='some operations based on erosion& dilate', figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.title('origin image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

bImg = binaryImg(img, threshold=200)
plt.subplot(2, 2, 2)
plt.title('binary image')
plt.imshow(bImg, cmap='gray')

edgeimg = edgedetect(bImg, 5)
plt.subplot(2, 2, 3)
plt.title('edge detect')
plt.imshow(edgeimg, cmap='gray')

cimg = closing(bImg, 5)
plt.subplot(2, 2, 4)
plt.title('closing')
plt.imshow(cimg, cmap='gray')
plt.show()
