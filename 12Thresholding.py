'''
reference:https://www.youtube.com/watch?v=ojapO75FV38&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=12&t=2009s
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt



# '''=========================global thresholding==========================='''
# def SelfOtus(img):
#     h, w = img.shape
#     global_P = np.zeros(shape=[256])
#     for i in range(h):
#         for j in range(w):
#             global_P[int(img[i, j])] += 1
#     global_P = global_P/(h*w)
#     intensity = np.arange(0, 256, 1)
#     global_mean = np.sum(np.multiply(intensity, global_P))
#     global_var = np.sum(np.multiply(np.power(intensity-global_mean, 2), global_P))
#
#     flag_ratio = 0
#     flag_intensity = 0
#     for i in range(len(intensity-1)):
#         p1 = np.sum(global_P[:i+1])
#         p2 = 1. - p1
#         m1 = np.sum(np.multiply(intensity[:i+1], global_P[:i+1])) / p1
#         m2 = np.sum(np.multiply(intensity[i+1:], global_P[i+1:])) / p2
#         part_var = np.sum(p1*np.power((m1-global_mean), 2))+np.sum(p2*np.power((m2-global_mean), 2))
#         ratio = part_var/global_var
#         if flag_ratio < ratio:
#             flag_ratio = ratio
#             flag_intensity = i
#
#     new_img = np.where(img > flag_intensity, 255, 0)
#     return new_img, flag_intensity
#
#
# fig, ax = plt.subplots(2, 2, figsize=(10, 10), num='global thresholding')
# img = cv2.imread('./data/IMG12.jpg')
# plt.subplot(221)
# plt.title('original img')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.subplot(222)
# plt.title('gray img')
# plt.imshow(gray_img, cmap='gray')
#
#
# intensity, cv2otsu_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
# plt.subplot(223)
# plt.title('otsu in opencv, intensity: '+str(intensity))
# plt.imshow(cv2otsu_img, cmap='gray')
#
#
# selfotsu_img, intensity = SelfOtus(gray_img)
# plt.subplot(224)
# plt.title('otsu by myself, intensity: '+str(intensity))
# plt.imshow(selfotsu_img, cmap='gray')
# plt.show()


'''adaptive thresholding======================================='''


def adaptiveThresholding(img, kernel_size):
    '''
    自适应阈值
    :param img: 图片矩阵
    :param kernel_size: 选择的邻域大小
    :return: 根据阈值确定的二值化图片
    '''
    h, w = img.shape
    border = int((kernel_size-1)/2)
    new_img = np.ones(shape=img.shape)*255
    intensity_list = np.arange(0, 256, 1)
    for i in range(h):
        for j in range(w):
            top = np.clip(i-border, 0, h-2)
            bottom = np.clip(i+border, 0, h-2)
            left = np.clip(j-border, 0, w-2)
            right = np.clip(j+border, 0, w-2)
            if left == right or top == bottom:
                pass
            else:
                P = np.zeros(shape=[256], dtype=np.int)
                for m in range(top, bottom+1):
                    for n in range(left, right+1):
                        P[int(img[m, n])] += 1
                P = P/((bottom-top+1)*(right-left+1))
                mean = np.sum(np.multiply(intensity_list, P))
                std_dev = np.sqrt(np.sum(np.multiply(np.power(intensity_list - mean, 2), P)))
                # 此处相当于调节阈值，原文中有多种方法，要选择适合自己的一种
                if img[i, j] < int(mean):
                    new_img[i, j] = 0
    return new_img


fig, ax = plt.subplots(1, 4, figsize=(12, 4), num='adaptive thresholding')
img = cv2.imread('./data/IMG13.jpg')
plt.subplot(141)
plt.title('original img')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(142)
plt.title('gray img')
plt.imshow(gray_img, cmap='gray')


adap_img = adaptiveThresholding(gray_img, 5)
plt.subplot(143)
plt.title('kernel = 5')
plt.imshow(adap_img, cmap='gray')

# adap_img = adaptiveThresholding(gray_img, 35)
# plt.subplot(144)
# plt.title('kernel = 35')
# plt.imshow(adap_img, cmap='gray')

plt.show()