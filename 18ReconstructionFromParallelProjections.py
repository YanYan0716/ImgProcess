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
    return result_num, result_img


def RadonTransform(img):
    '''
    中文名为雷登变换 reference:https://blog.csdn.net/jasneik/article/details/115099488
    :param img:
    :param theta:
    :return:
    '''
    [H, W] = img.shape
    rotate_center = (W / 2, H / 2)
    result = np.zeros(shape=img.shape, dtype=np.float32)
    angles = np.linspace(0, 180, H)

    for angle in angles:
        M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0).astype(np.float32)
        new_img = cv2.warpAffine(img, M, (W, H))  # 图像的旋转
        result[:, np.int(angle*H/180)-1] = np.sum(new_img, axis=0)
    return result


def InvertRadonTransform(RadonImg):
    '''
    反雷登变换 reference:https://blog.csdn.net/jasneik/article/details/115099488
    :param RadonImg:雷登变换后的图像
    :return:
    '''
    [H, W] = RadonImg.shape
    rotate_center = (W / 2, H / 2)
    result = np.zeros(shape=RadonImg.shape, dtype=np.float32)
    angles = np.linspace(0, 180, H)

    for angle in angles:
        img_ = np.zeros(shape=RadonImg.shape, dtype=np.float) #生成的中间图像
        for i in range(W):
            img_[i, :] = RadonImg[:, np.int(angle*H/180)-1]
        M = cv2.getRotationMatrix2D(rotate_center, -angle, 1.0).astype(np.float32)
        img_ = cv2.warpAffine(img_, M, (W, H))
        result += img_
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

result = RadonTransform(gray_img)
result = result*255 / (np.max(result)-np.min(result))
plt.subplot(2, 3, 4)
plt.title('radon transform from original')
plt.imshow(np.array(result, dtype=np.uint), cmap='gray')
plt.show()


inverse_result = InvertRadonTransform(result)
inverse_result = inverse_result*255 / (np.max(inverse_result)-np.min(inverse_result))
plt.subplot()
plt.title('invert radon transform')
plt.imshow(np.array(inverse_result, dtype=np.uint), cmap='gray')
plt.show()