'''
reference:https://www.youtube.com/watch?v=ZF-3aORwEc0&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=13
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



'''========basic region growing =================='''
def PixelWithThreshold(img_array, img_pixel, threshold, position):
    '''
    根据某中心像素值，判断其相邻8个像素点是否具有相同颜色
    :param img_array: 图像的数组形式
    :param img_pixel: 中心像素值
    :param threshold: 阈值，大于该阈值则证明两者不相似
    :param position: 中心像素的坐标位置 [h, w]
    :return: 列表，在阈值范围内的像素坐标
    '''
    [H, W] = position

    # 找到center pixel的8个相邻点
    if len(img_array.shape) == 3:
        h, w, c = img_array.shape
    else:
        h, w = img_array.shape
    top, bottom = np.clip(H - 1, 0, h - 1), np.clip(H + 1, 0, h - 1)
    left, right = np.clip(W - 1, 0, w - 1), np.clip(W + 1, 0, w - 1)
    lefttop, ttop, righttop = [top, left], [top, W], [top, right]
    lleft, rright = [H, left], [H, right]
    leftbottom, bbottom, rightbottom = [bottom, left], [bottom, W], [bottom, right]
    neighbors = [lefttop, ttop, righttop, lleft, rright, leftbottom, bbottom, rightbottom]
    neighbors_list = list()

    # 判断8个相邻点中在阈值范围内的即需要的，不在阈值范围内的即要舍弃的
    for neigh in neighbors:
        # 如果是RGB三通道
        if len(img_array.shape) == 3:
            if abs(np.int32(img_array[neigh[0], neigh[1], 0]) - img_pixel[0]) < threshold and \
                    abs(np.int32(img_array[neigh[0], neigh[1], 1]) - img_pixel[1]) < threshold and \
                    abs(np.int32(img_array[neigh[0], neigh[1], 2]) - img_pixel[2]) < threshold:
                neighbors_list.append(neigh)
            else:
                pass
        # 如果是灰度图单通道
        else:
            a = abs(np.int32(img_array[neigh[0], neigh[1]])-img_pixel)
            if a < threshold:
                neighbors_list.append(neigh)
            else:
                pass
    return neighbors_list


def RegionWithThreshold(img_array, img_pixel, threshold, position):
    '''
    根据阈值进行imagesegmentation,
    :param img_array: 图像的数组形式
    :param img_pixel: 中心点像素值
    :param threshold: 阈值范围，范围越大则选取的区域越大
    :param position: 中心点像素的位置
    :return: list(), 在阈值范围内，并且和中心点坐标相连的像素坐标 [h, w]
    '''
    last_list = list()
    now_list = [position]
    while last_list != now_list:
        old_list = list(now_list)
        extra_list = list()
        for point in now_list:
            if point not in last_list:
                neighbor_points = PixelWithThreshold(img_array, img_pixel, threshold, point)
                extra_list += neighbor_points

        new_list = now_list + extra_list
        for i in new_list:
            if not i in now_list:
                now_list.append(i)
        last_list = list(old_list)
    return now_list


def click(event):
    '''
    鼠标点击事件，获取点击的坐标位置和相对应的像素值 进行相应的阈值选择处理
    :param event:
    '''
    (w, h) = event.xdata, event.ydata
    click_position = [int(h), int(w)]
    img_array = np.array(img)
    img_pixel = img_array[click_position[0], click_position[1]]
    points_list = RegionWithThreshold(img_array, img_pixel, threshold=20, position=click_position)
    for i in range(len(points_list)):
        if len(img_array.shape) == 3:
            img_array[points_list[i][0], points_list[i][1]] = [0, 0, 255]
        else:
            img_array[points_list[i][0], points_list[i][1]] = 0
    plt.figure()
    plt.subplot()
    plt.title('image segmentation')
    plt.imshow(img_array, cmap='gray')
    plt.show()


img = cv2.imread('./data/IMG14.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fig = plt.figure(num='origin image')
cid = fig.canvas.mpl_connect('button_press_event', click)
plt.title('please pick a position from the image, a large range will take more time')
plt.imshow(img, cmap='gray')
plt.show()
