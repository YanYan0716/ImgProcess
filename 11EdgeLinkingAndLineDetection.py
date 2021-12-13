'''
reference:https://www.youtube.com/watch?v=_con6DnhkaA&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=11
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


def earn8Direction(b, c):
    '''
    当我们知道当前坐标b与其对应的c时，找出b的8个位置的临近坐标，以c为8个位置的起始点
    :param b: edge point
    :param c: 与edge配对的前一个坐标，详细含义参照原视频中的c0， c1……
    :return: 8个相邻坐标的列表
    '''
    ltop = [b[0] - 1, b[1] - 1]
    top = [b[0] - 1, b[1]]
    rtop = [b[0] - 1, b[1] + 1]
    right = [b[0], b[1] + 1]

    rbottom = [b[0] + 1, b[1] + 1]
    bottom = [b[0] + 1, b[1]]
    lbottom = [b[0] + 1, b[1] - 1]
    left = [b[0], b[1] - 1]

    res = [ltop, top, rtop, right, rbottom, bottom, lbottom, left]
    flag = 0
    while res[flag] != c:
        flag = flag + 1
    res = res[flag + 1:] + res[:flag + 1]
    return res


def edgelinking(img, start_position):
    '''
    边缘检测的简单实现，所形成的图中绿色为检测到的边缘点，红色为边缘点的前一个point（详细解释请见元视频）
    :param img: cv2读取的image
    :param start_position: edge points的最开始的一个坐标
    :return: 边缘检测完成的边缘image
    '''
    final_img = np.array(img)
    plt.ion()
    plt.figure(num="edge detection's processing")
    [bx, by] = start_position
    [cx, cy] = bx, by - 1
    img_ = np.array(img)
    img_[bx, by, :] = [0, 255, 0]
    img_[cx, cy, :] = [255, 0, 0]
    final_img[bx, by, :] = [0, 255, 0]
    plt.clf()
    plt.imshow(img_)
    plt.pause(0.5)

    neighbor = earn8Direction([bx, by], [cx, cy])
    position_point = 0
    while img[neighbor[position_point][0], neighbor[position_point][1], 0] < 251:
        position_point += 1

    [bx, by] = neighbor[position_point]
    [cx, cy] = neighbor[position_point - 1]
    img_ = np.array(img)
    img_[bx, by, :] = [0, 255, 0]
    img_[cx, cy, :] = [255, 0, 0]
    final_img[bx, by, :] = [0, 255, 0]
    plt.clf()
    plt.imshow(img_)
    plt.pause(0.5)
    while [bx, by] != start_position:
        neighbor = earn8Direction([bx, by], [cx, cy])
        position_point = 0
        while img[neighbor[position_point][0], neighbor[position_point][1], 0] < 251:
            position_point += 1

        [bx, by] = neighbor[position_point]
        [cx, cy] = neighbor[position_point - 1]
        img_ = np.array(img)
        img_[bx, by, :] = [0, 255, 0]
        img_[cx, cy, :] = [255, 0, 0]
        final_img[bx, by, :] = [0, 255, 0]
        plt.clf()
        plt.imshow(img_)
        plt.pause(0.5)
    plt.ioff()
    plt.show()
    return final_img


img = cv2.imread('./data/IMG11.jpg')
start_position = [2, 3]
res_img = edgelinking(img, start_position)
plt.title('final edge detiction result')
plt.imshow(res_img)
plt.show()


'''======================hough transform==========='''


def edge(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


import seaborn as sns
import pandas as pd


def MyHough(img, rho, theta, threshold):
    '''
    手动实现霍夫变换
    :param img: 边缘检测后的img
    :param rho: 距离原点的距离
    :param theta: 将角度分为theta份
    :param threshold: 要检测出多少个点来就可以形成一条直线
    :return: 直线列表极坐标形式, theta的间隔取值， rho对应的结果
    '''
    (h, w) = img.shape
    lines = list()
    theta_list = np.arange(-np.pi/2, np.pi/2, np.pi / theta)
    rho = int(np.max([h, w])/rho)

    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:
                line = list()
                for self_theta in theta_list:
                    self_rho = i * np.sin(self_theta) + j * np.cos(self_theta)
                    line.append(self_rho)
                lines.append(line)
    final_result = list()
    lines_array = np.array(lines)
    for col in range(lines_array.shape[1]):
        result = pd.cut(lines_array[:, col], rho)
        values = result.value_counts().values
        for ind in range(len(values)):
            if values[ind] > threshold:
                final_result.append((theta_list[col], result.value_counts().keys()[ind].mid))
    return final_result, theta_list, lines_array


fig, ax = plt.subplots(2, 2, figsize=(10, 10), num='canny')
img = cv2.imread('./data/IMG10.jpg')
plt.subplot(221)
plt.title('original img')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

BW = edge(img)
plt.subplot(222)
plt.title('canny edge detection')
plt.imshow(BW, cmap='gray')

img_ = np.array(BW)
lines = cv2.HoughLines(img_, 0.8, np.pi / 90, 100)
img2 = cv2.imread('./data/IMG10.jpg')
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255))
plt.subplot(223)
plt.title('hough with cv2')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))


img2 = cv2.imread('./data/IMG10.jpg')
result, thetalist, rholist = MyHough(BW, 0.8, 90, 100)
for res in result:
    (t, r) = res
    a = np.cos(t)
    b = np.sin(t)
    x0 = a * r
    y0 = b * r
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255))
plt.subplot(224)
plt.title('hough with myself')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.show()


# 直线簇转换到极坐标下的表示
for i in range(rholist.shape[0]):
    plt.plot(thetalist, rholist[i])
    break
plt.show()

