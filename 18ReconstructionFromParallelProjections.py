"""
reference:https://www.youtube.com/watch?v=ZgcD4C-4u0Q&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=21
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def RadonTransform_example(img, angle):
    '''
    RadonTransform的一个小例子，显示投影一次对应的曲线图是什么样子的
    :param img:
    :param angle:
    :return:
    '''
    [H, W] = img.shape
    result_img = np.zeros(shape=img.shape)
    rotate_center = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    img = cv2.warpAffine(img, M, (W, H))
    [H, W] = img.shape
    result_num = list()
    for i in range(W):
        result_num.append(np.sum(img[:, i]))
    for j in range(H):
        result_img[j, :] = np.array(result_num)
    result_img = result_img*255/(np.max(result_img)-np.min(result_img))
    result_img = cv2.warpAffine(result_img, M, (W, H))
    return result_num, result_img


def RadonTransform(img, theta):
    '''

    :param img:
    :param theta:
    :return:
    '''
    [H, W] = img.shape
    rotate_center = (W / 2, H / 2)
    result = np.zeros(shape=img.shape)

    for angle in range(0, 180, theta):
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        new_img = cv2.warpAffine(img, M, (W, H))  # 图像的旋转
        # 获得res
        res = np.zeros(shape=img.shape)
        result_num = list()
        for i in range(W):
            result_num.append(np.sum(new_img[:, i]))
        for j in range(H):
            res[j, :] = np.array(result_num)
        res = res * 255 / (np.max(res) - np.min(res))
        new_res = cv2.warpAffine(res, M, (W, H))  # 图像的旋转
        result += new_res


    return result


figure = plt.figure(num='ImageRestorationAndTheWienerFilter', figsize=(14, 7))
img = cv2.imread('./data/IMG22.jpg')
plt.subplot(2, 3, 1)
plt.title('origin')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
result_num, result_img = RadonTransform_example(gray_img, 0)
plt.subplot(2, 3, 2)
plt.title('radon transform from W')
plt.plot(result_num)
plt.subplot(2, 3, 5)
plt.title('radon transform from W')
plt.imshow(np.array(result_img, dtype=np.uint), cmap='gray')


result_num, result_img = RadonTransform_example(gray_img, 90)
plt.subplot(2, 3, 3)
plt.title('radon transform from H')
plt.plot(result_num)
plt.subplot(2, 3, 6)
plt.title('radon transform from H')
plt.imshow(np.array(result_img, dtype=np.uint), cmap='gray')

# result = RadonTransform(gray_img, 1)
# plt.subplot(2, 3, 4)
# plt.title('123')
# plt.imshow(np.array(result, dtype=np.uint), cmap='gray')

plt.show()
