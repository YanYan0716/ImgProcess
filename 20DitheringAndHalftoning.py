'''
reference:https://www.youtube.com/watch?v=UJtV3DdjCVY&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=23&t=2s
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


def Dithering(img, threshold=128, A=64):
    '''
    图像抖动算法
    :param img:
    :param threshold:
    :param A:
    :return:
    '''
    img_array = np.array(img)
    noise = np.random.uniform(-A, A, size=img_array.shape)
    img_array = img_array + noise
    new_img = np.zeros(shape=img_array.shape)
    (H, W) = img_array.shape
    for h in range(H):
        for w in range(W):
            new_img[h, w] = 0 if img_array[h, w] < threshold else 255
    return new_img


def Halftoning(img):
    [H, W] = img.shape
    new_img = np.zeros(shape=img.shape)
    kernel = np.array([[40, 60, 150, 90, 10],
                       [80, 170, 240, 200, 110],
                       [140, 210, 250, 220, 130],
                       [120, 190, 230, 180, 70],
                       [20, 100, 160, 50, 30]])
    for h in range(0, H, 5):
        for w in range(0, W, 5):
            new_img[h:h + 5, w:w + 5] = np.array(img[h:h + 5, w:w + 5] + kernel > 255, dtype=np.int)
    return new_img


img = cv2.imread('./data/IMG23.jpg')
figure = plt.figure(num='dither', figsize=(9, 3))
plt.subplot(1, 4, 1)
plt.title('original image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.subplot(1, 4, 2)
plt.title('gray image')
plt.imshow(gray_img, cmap='gray')

result = Dithering(gray_img)
plt.subplot(1, 4, 3)
plt.title('dithering image')
plt.imshow(result, cmap='gray')

result = Halftoning(gray_img)
plt.subplot(1, 4, 4)
plt.title('halften image')
plt.imshow(result, cmap='gray')

plt.show()
