'''
reference:https://www.youtube.com/watch?v=O2RwWHWHQlM&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX
'''

'''
reference:https://www.youtube.com/watch?v=UJtV3DdjCVY&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=23&t=2s
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


def SpatialWatermark(img1, img2, alpha):
    new_img = (1 - alpha) * img1 + alpha * img2
    new_img = new_img.astype(img1.dtype)
    return new_img


def LeastSignifitionBits(original, copyright, bits=1):
    '''
    reference:https://zhuanlan.zhihu.com/p/89765000
    :param img:
    :param bits:
    :return:
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
plt.show()
