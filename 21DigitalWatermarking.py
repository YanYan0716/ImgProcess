'''
reference:https://www.youtube.com/watch?v=O2RwWHWHQlM&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX
reference:https://www.youtube.com/watch?v=UJtV3DdjCVY&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=23&t=2s
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random


def SpatialWatermark(img1, img2, alpha):
    new_img = (1 - alpha) * img1 + alpha * img2
    new_img = new_img.astype(img1.dtype)
    return new_img


def LeastSignifitionBits(original, copyright, bits=1):
    '''
    reference:https://zhuanlan.zhihu.com/p/89765000
    '''
    original = original.astype(np.uint8)
    copyright = copyright.astype(np.uint8)

    watermark = original.copy()
    copyright_ = copyright.copy()

    copyright_[copyright_ < 200] = 1
    copyright_[copyright_ >= 200] = 0

    for i in range(0, watermark.shape[0]):
        for j in range(0, watermark.shape[1]):
            watermark[i, j, :] = (watermark[i, j, :] // (2 * bits)) * (2 * bits)

    for i in range(0, copyright_.shape[0]):
        for j in range(0, copyright_.shape[1]):
            watermark[i, j, :] = watermark[i, j, :]
    return watermark


def LeastSignifitionBits_decoder(watermark):
    watermark = watermark.astype(np.uint8)
    watermark = (watermark % 2) * 255
    return watermark


def SimpleFrequencyFlipping(original, copyright, alpha=10):
    original = original / 255
    copyright = copyright / 255
    ori_h, ori_w, ori_c = original.shape
    copy_h, copy_w, copy_c = copyright.shape

    ori_f = np.fft.fft2(original)
    ori_f = np.fft.fftshift(ori_f)

    # 水印图像编码
    x, y = list(range(math.floor(ori_h / 2))), list(range(ori_w))
    random.seed(ori_h + ori_w)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(original.shape)

    for i in range(math.floor(ori_h / 2)):
        for j in range(ori_w):
            if x[i] < copy_h and y[j] < copy_w:
                tmp[i][j] = copyright[x[i]][y[j]]
                tmp[ori_h - i - 1][ori_w - j - 1] = tmp[i][j]

    res_f = ori_f + alpha * tmp
    res = np.fft.ifftshift(res_f)
    res = np.abs(np.fft.ifft2(res)) * 255
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res


def SimpleFrequencyFlipping_decoder(original, watermark, alpha=10):
    '''

    :param original: 原图像
    :param watermark: 水印后的图像
    :param alpha:
    :return:
    '''
    original = original/255
    watermark = watermark/255
    ori_h, ori_w, ori_c = original.shape

    ori_f = np.fft.fft2(original)
    ori_f = np.fft.fftshift(ori_f)
    water_f = np.fft.fft2(watermark)
    water_f = np.fft.fftshift(water_f)
    mark = np.abs((water_f-ori_f)/alpha)
    res = np.zeros(original.shape)

    # 获取随机种子
    x, y = list(range(math.floor(ori_h/2))), list(range(ori_w))
    random.seed(ori_h+ori_w)
    random.shuffle(x)
    random.shuffle(y)
    for i in range(math.floor(ori_h/2)):
        for j in range(ori_w):
            res[x[i]][y[j]] = mark[i][j]*255
            res[ori_h-i-1][ori_w-j-1] = res[i][j]
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res


img1 = cv2.imread('./data/IMG24.jpg')
figure = plt.figure(num='dither', figsize=(9, 7))
plt.subplot(2, 3, 1)
plt.title('image1')
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

img2 = cv2.imread('./data/IMG25.jpg')
plt.subplot(2, 3, 2)
plt.title('image2')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

result = SpatialWatermark(img1, img2, alpha=0.3)
plt.subplot(2, 3, 3)
plt.title('SpatialWatermark')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

result = LeastSignifitionBits(img1, img2, bits=1)
img1_ = LeastSignifitionBits_decoder(result)
if img1.all() == img1_.all():
    print('used function LeastSignifitionBits_decoder to return the watermark to original')
plt.subplot(2, 3, 4)
plt.title('make the least significant bits')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

result = SimpleFrequencyFlipping(img1, img2, 10)
plt.subplot(2, 3, 5)
plt.title('simple frequency flipping encode')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

result = SimpleFrequencyFlipping_decoder(img1, result, 10)
plt.subplot(2, 3, 6)
plt.title('simple frequency flipping decode')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()
