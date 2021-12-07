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
    ltop = [b[0]-1, b[1]-1]
    top = [b[0]-1, b[1]]
    rtop = [b[0]-1, b[1]+1]
    right = [b[0], b[1]+1]

    rbottom = [b[0]+1, b[1]+1]
    bottom = [b[0]+1, b[1]]
    lbottom = [b[0]+1, b[1]-1]
    left = [b[0], b[1]-1]

    res = [ltop, top, rtop, right, rbottom, bottom, lbottom, left]
    flag = 0
    while res[flag] != c:
        flag = flag+1
    res = res[flag+1:]+res[:flag+1]
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
    [cx, cy] = bx, by-1
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
    [cx, cy] = neighbor[position_point-1]
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