'''
直方图与点操作
reference：https://www.youtube.com/watch?v=qKWPBzRD-U0&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=4&t=3s
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = cv2.imread('./data/IMG1.jpg')
print(img.shape)
# cv2.imshow('origin', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''颜色分布直方图=================================================='''
plt.figure('rgb histogram', figsize=(10, 10))
# RGB颜色分布直方图
ax1 = plt.subplot(221)
ax1.hist(img[:, :, 0].ravel(), bins=255, color='b')
ax2 = plt.subplot(222)
ax2.hist(img[:, :, 1].ravel(), bins=255, color='g')
ax3 = plt.subplot(223)
ax3.hist(img[:, :, 2].ravel(), bins=255, color='r')
plt.subplot(224)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 使用numpy生成RGB颜色直方图
fig, ax = plt.subplots(2, 2, figsize=(10, 10), num='rgb histogram from numpy')
colors = ['b', 'g', 'r']
for i in range(3):
    hist, bins = np.histogram(img[:, :, i].ravel(), bins=256, range=(0, 256))
    ax[i // 2, i % 2].plot(0.5 * (bins[:-1] + bins[1:]), hist, label=colors[i], color=colors[i])
plt.subplot(224)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 灰度图颜色分布直方图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure('gray histogram', figsize=(10, 5))
ax1 = plt.subplot(121)
ax1.hist(gray_img.ravel(), bins=255, color='black')
plt.subplot(122)
plt.imshow(gray_img, cmap='gray')
plt.show()

'''二值图生成=================================================='''
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)


cv2.imshow('thresh', th)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 手写代码实现二值图
def EarnBinaryImg(img, threshold):
    '''
    手写代码实现二值图
    :param img: 灰度图数组 from cv2.imread
    :param threshold: 二值图的阈值
    :return: 二值图数组
    '''
    img = np.clip(img, 0, 255)
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i][j] >= threshold:
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary_img = EarnBinaryImg(gray_img, 127)
plt.figure('binary image', figsize=(10, 5))
plt.subplot(121)
plt.imshow(th, cmap='gray')
plt.title('binary image from opencv')
plt.subplot(122)
plt.imshow(binary_img, cmap='gray')
plt.title('binary image from self function')
plt.show()

'''image diginal negative=================================================='''


# for grayscale image
def ImageNeg(img):
    '''
    实现像素值取反
    :param img: np.array from cv2
    :return: 像素取反后的图像数组
    '''
    img = np.clip(img, 0, 255)
    if len(img.shape) == 2:
        height, width = img.shape
        for i in range(height):
            for j in range(width):
                img[i][j] = 255 - img[i][j]
    else:
        height, width, channel = img.shape
        for i in range(height):
            for j in range(width):
                for c in range(channel):
                    img[i][j][c] = 255 - img[i][j][c]

    return img


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_neg = ImageNeg(gray_img)
rgb_neg = ImageNeg(img)
plt.figure('image negative', figsize=(10, 10))
plt.subplot(221)
plt.imshow(gray_img, cmap='gray')
plt.title('gray image')
plt.subplot(222)
plt.imshow(gray_neg, cmap='gray')
plt.title('gray negative image')
plt.subplot(223)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('rgb image')
plt.subplot(224)
plt.imshow(cv2.cvtColor(rgb_neg, cv2.COLOR_BGR2RGB))
plt.title('rgb negative image')
plt.show()

'''contrast enhancement==========================='''


def ContrastEnhancement(img, alpha, beta):
    '''
    手动实现对比度增强的效果 符合线性函数y=ax+b，x即为像素点的值
    :param img: 数组
    :param alpha: a
    :param betha: b
    :return: 经过对比度增强后的图像数组
    '''
    if len(img.shape) == 2:
        height, width = img.shape
        for i in range(height):
            for j in range(width):
                img[i][j] = np.clip((alpha * (img[i][j]) + beta), 0, 255)
    else:
        height, width, channel = img.shape
        for i in range(height):
            for j in range(width):
                for c in range(channel):
                    img[i][j][c] = np.clip((alpha * (img[i][j][c]) + beta), 0, 255)
    return img

# 灰度图的对比度增强
fig, ax = plt.subplots(3, 2, figsize=(10, 10), num='contrast enhancement about gray')
img = cv2.imread('./data/IMG2.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(321)
plt.imshow(gray_img, cmap='gray')
gray_hist, gray_bins = np.histogram(gray_img[:, :].ravel(), bins=256, range=(0, 256))
ax[0, 1].plot(0.5 * (gray_bins[:-1] + gray_bins[1:]), gray_hist, label='gray', color='black')

gray_img_ = cv2.convertScaleAbs(gray_img, alpha=2.5, beta=0)  # 满足线性映射y=ax+b
plt.subplot(323)
plt.imshow(gray_img_, cmap='gray')
gray_hist_, gray_bins_ = np.histogram(gray_img_[:, :].ravel(), bins=256, range=(0, 256))
ax[1, 1].plot(0.5 * (gray_bins_[:-1] + gray_bins_[1:]), gray_hist_, label='gray_enhence', color='black')

gray_img_self = ContrastEnhancement(gray_img, alpha=2.5, beta=0)  # 满足线性映射y=ax+b
plt.subplot(325)
plt.imshow(gray_img_self, cmap='gray')
gray_hist_self, gray_bins_self = np.histogram(gray_img_self[:, :].ravel(), bins=256, range=(0, 256))
ax[2, 1].plot(0.5 * (gray_bins_self[:-1] + gray_bins_self[1:]), gray_hist_self, label='gray_enhence_self', color='black')
plt.show()

# rgb的对比度增强
fig, ax = plt.subplots(1, 3, figsize=(10, 3), num='contrast enhancement about rgb')
img = cv2.imread('./data/IMG2.jpg')
plt.subplot(131)
plt.title('origin')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img_ = cv2.convertScaleAbs(img, alpha=2.5, beta=0)  # 满足线性映射y=ax+b
plt.subplot(132)
plt.title('ContrastEnhancement from cv2')
plt.imshow(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
img_self = ContrastEnhancement(img, alpha=2.5, beta=0)  # 满足线性映射y=ax+b
plt.subplot(133)
plt.title('ContrastEnhancement from myself')
plt.imshow(cv2.cvtColor(img_self, cv2.COLOR_BGR2RGB))
plt.show()


'''histogram equalization==========================='''


def HistEqua(img):
    height, width = img.shape
    hist, bins = np.histogram(img[:, :].ravel(), bins=256, range=(0, 256))
    hist_ = np.array(hist)
    for i in range(1, len(hist)):
        for j in range(i):
            hist_[i] += hist[j]
        hist_[i] = (hist_[i]/(height*width))*255

    for i in range(height):
        for j in range(width):
            img[i][j] = hist_[img[i][j]]
    return img


fig, ax = plt.subplots(1, 3, figsize=(10, 3), num='histogram equalization')
img = cv2.imread('./data/IMG2.jpg')
plt.subplot(131)
plt.title('origin')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# 在亮度上实现直方图均衡化
img_ = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
img_[:, :, 0] = cv2.equalizeHist(img_[:, :, 0])
img_ = cv2.cvtColor(img_, cv2.COLOR_YCrCb2RGB)
plt.subplot(132)
plt.title('hist equal from cv2')
plt.imshow(img_)

img_self = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
img_self[:, :, 0] = HistEqua(img_self[:, :, 0])
plt.subplot(133)
plt.title('hist equal from myself')
plt.imshow(cv2.cvtColor(img_self, cv2.COLOR_YCrCb2RGB))
plt.show()


# filter

def ImgFilter(img, filter):
    height, width = img.shape
    img_ = np.zeros(shape=(height, width), dtype=np.uint8)
    for i in range(height-1):
        for j in range(width-1):
            left = i-1
            left = np.clip(left, 0, height-3)
            right = left+3
            top = j-1
            top = np.clip(top, 0, width-3)
            bottom = top+3
            img_[i, j] = int(np.sum(img[left:right, top:bottom]*filter))
    return img_


img = cv2.imread('./data/IMG1.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


filter = np.array([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]])
# filter = np.array([[0.11, 0.11, 0.11], [0.11, 0.11, 0.11], [0.11, 0.11, 0.11]])

img = ImgFilter(gray_img, filter)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img_ = Image.fromarray(img)
plt.imshow(img_, cmap='gray')
plt.show()
# cv2.imshow('sd', img)