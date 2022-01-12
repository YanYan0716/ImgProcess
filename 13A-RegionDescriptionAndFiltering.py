'''
reference:https://www.youtube.com/watch?v=_kwZj-EB1OU&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=16
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

'''===============distance Transform======================'''


# reference:https://blog.csdn.net/a8039974/article/details/78931047
def distancetransform(binary_img):
    '''
    二值图的距离转换
    将图像中的目标像素点分类，分为内部点，外部点和孤立点。
    以中心像素的四邻域为例，如果中心像素为目标像素(值为1)且四邻域都为目标像素(值为1)，则该点为内部点。
    如果该中心像素为目标像素，四邻域为背景像素(值为0)，则该中心点为孤立点 即为kernel的来源
    :param binary_img: 二值图，其中目标物为255 背景为0
    :return:
    '''
    [H, W] = binary_img.shape
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    inter_list = list()
    uninter_list = list()
    for h in range(1, H - 2):
        for w in range(1, W - 2):
            if binary_img[h][w] == 255:
                # 计算图像中所有的内部点和非内部点，点集分别为inter_list uninter_list
                if np.sum(binary_img[h - 1:h + 2, w - 1:w + 2] * kernel) == 5 * 255:
                    inter_list.append([h, w])
                else:
                    uninter_list.append([h, w])
    # 对于每一个内部点(x,y)，使用距离公式disf()计算在uninter_list中的最小距离，这些最小距离构成集合res
    # 其中距离公式可根据实际场景进行修改，此处使用的是欧式距离
    inter = np.array(inter_list)
    uninter = np.array(uninter_list)
    inter_x, inter_y = inter[:, 0], inter[:, 1]
    uninter_x, uninter_y = uninter[:, 0], uninter[:, 1]

    [a1, a2] = np.meshgrid(inter_x, uninter_x)
    [b1, b2] = np.meshgrid(inter_y, uninter_y)
    x_power = np.power((a1 - a2), 2)
    y_power = np.power((b1 - b2), 2)
    res = np.min(np.sqrt((x_power + y_power)), axis=0)
    # 计算res中的最大最小值Max, Min
    min_ = np.min(res)
    max_ = np.max(res)

    res_img = binary_img.copy()
    # 对于每一个内部点，转换后的灰度值G计算
    for i in range(len(inter_list)):
        res_img[inter_list[i][0], inter_list[i][1]] = ((res[i] - min_) / (max_ - min_)) * 255

    # 边缘点黑化
    for i in range(len(uninter_list)):
        res_img[uninter_list[i][0], uninter_list[i][1]] = 0
    return res_img


img = cv2.imread('./data/IMG18.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
binary_img = 255 - binary_img
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)

plt.figure(figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.title('origin img')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title('binary img')
plt.imshow(binary_img, cmap='gray')

dist_trans = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
plt.subplot(2, 2, 3)
plt.title('distTrans from cv2')
plt.imshow(dist_trans, cmap='gray')

res = distancetransform(binary_img)
plt.subplot(2, 2, 4)
plt.title('distTrans from myself')
plt.imshow(res, cmap='gray')
plt.show()


'''====================骨骼提取算法=================='''

def f1(img, border_list, border_kernel, threshold_list):
    border_list_ = border_list.copy()
    for i in range(len(border_list)):
        weight = np.sum(
            img[border_list[i][0] - 1:border_list[i][0] + 2, border_list[i][1] - 1:border_list[i][1] + 2]
            * border_kernel/255)
        if weight in threshold_list:
            border_list_.remove(border_list[i])
            img[border_list[i][0], border_list[i][1]] = 0
    border_list = border_list_.copy()
    return img, border_list


def make_border(img, border_kernel):
    img = img/255.
    [H, W] = img.shape
    border_list = []
    A0 = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96,
          112, 120, 124, 126, 127, 129, 131, 135, 143, 159, 191, 192,
          193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241,
          243, 247, 248, 249, 251, 252, 253, 254]

    for h in range(1, H - 2):
        for w in range(1, W - 2):
            if img[h, w] == 1.:
                weight = np.sum(img[h - 1:h + 2, w - 1:w + 2] * border_kernel)
                if weight in A0:
                    border_list.append([h, w])

    return border_list

def K3M(binary_img):
    '''
    图像的骨骼提取算法K3M，参考文献http://matwbn.icm.edu.pl/ksiazki/amc/amc20/amc2029.pdf
    :param binary_img: 设定背景为0 前景为255
    :return: 只有骨骼信息的二值图
    '''
    [H, W] = binary_img.shape

    # 获取border pixels，border pixel的8个邻居必定不全为白
    border_kernel = np.ones(shape=(3, 3))
    border_kernel = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]])
    border_list = make_border(binary_img, border_kernel)

    while True:
        old_border_list = border_list.copy()
        A1 = [7, 14, 28, 56, 112, 131, 193, 224]
        binary_img, border_list = f1(binary_img, border_list, border_kernel, A1)

        A2 = [7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135, 193, 195, 224, 225, 240]
        binary_img, border_list = f1(binary_img, border_list, border_kernel, A2)

        A3 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120, 124, 131, 135,
              143, 193, 195, 199, 224, 225, 227, 240, 241, 248]
        binary_img, border_list = f1(binary_img, border_list, border_kernel, A3)

        A4 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120,
                124, 126, 131, 135, 143, 159, 193, 195, 199, 207,
                224, 225, 227, 231, 240, 241, 243, 248, 249, 252]
        binary_img, border_list = f1(binary_img, border_list, border_kernel, A4)

        A5 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120,
                124, 126, 131, 135, 143, 159, 191, 193, 195, 199,
                207, 224, 225, 227, 231, 239, 240, 241, 243, 248,
                249, 251, 252, 254]
        binary_img, border_list = f1(binary_img, border_list, border_kernel, A5)
        border_list = make_border(binary_img, border_kernel)
        if len(old_border_list) == len(border_list):
            break

    A1pix = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56,
             60, 62, 63, 96, 112, 120, 124, 126, 127, 129, 131,
             135, 143, 159, 191, 192, 193, 195, 199, 207, 223,
             224, 225, 227, 231, 239, 240, 241, 243, 247, 248,
             249, 251, 252, 253, 254]
    binary_img, border_list = f1(binary_img, border_list, border_kernel, A1pix)
    return binary_img


img = cv2.imread('./data/IMG19.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
binary_img = 255 - binary_img

plt.figure(figsize=(7, 5))
plt.subplot(1, 3, 1)
plt.title('origin img')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('binary img')
plt.imshow(binary_img, cmap='gray')

skele = K3M(binary_img)
plt.subplot(1, 3, 3)
plt.title('K3M')
plt.imshow(skele, cmap='gray')
plt.show()
