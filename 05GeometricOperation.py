'''
几何变换
reference：https://www.youtube.com/watch?v=Gu9mSHwI3ec&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=5
'''

import cv2
import numpy as np
import matplotlib.pylab as plt
from PIL import Image


def ForRotate(img, degree):
    '''
    forwards mapping
    :param img:
    :param degree: from 0 to 90
    :return:
    '''
    height, width, channels = img.shape
    degree = degree*np.pi/180
    # 计算旋转后图形的高 宽
    new_height = int(height*np.cos(degree)+width*np.sin(degree))
    new_width = int(height*np.sin(degree)+width*np.cos(degree))
    new_img = np.zeros(shape=(new_height, new_width, channels), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            old_x, old_y = j-(width/2), i-(height/2)
            new_x = int(old_x*np.cos(degree)-old_y*np.sin(degree))
            new_y = int(old_x*np.sin(degree)+old_y*np.cos(degree))
            new_x = np.clip(int(new_x + (new_width/2)), 0, new_width-1)
            new_y = np.clip(int(new_y + (new_height/2)), 0, new_height-1)
            new_img[new_y, new_x, :] = img[i, j, :]
    return new_img


img = cv2.imread('./data/IMG1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = ForRotate(img, 30)
print(img.shape)
plt.imshow(img)
plt.show()


from interval import Interval
def BilinearInterpolation(img, x, y, h_range, w_range):
    '''
    双线性插值
    :param img: 图片矩阵
    :param x: img的x轴坐标，float
    :param y: img的y轴坐标，float
    :param h_range: img高的取值范围
    :param w_range: img宽的取值范围
    :return: 所需颜色值
    '''
    if (x in w_range) and (y in h_range):
        alpha = x - int(x)
        beta = y - int(y)
        x1 = img[int(y), int(x), :]
        y1 = img[int(y), np.clip(int(x+1), 0, img.shape[1]-1), :]
        x2 = img[np.clip(int(y+1), 0, img.shape[0]-1), int(x), :]
        y2 = img[np.clip(int(y+1), 0, img.shape[0]-1), np.clip(int(x+1), 0, img.shape[1]-1), :]
        color = (1-alpha)*(1-beta)*x1+(1-beta)*alpha*y1+(1-alpha)*beta*x2+alpha*beta*y2
        color = np.array(color, dtype=np.int)
    else:
        color = np.array([0, 0, 0], dtype=np.int)
    return color


def BackRotate(img, degree, bilinear_interpolation=False):
    '''
    backword mapping
    :param img:
    :param degree: 旋转角度0到90
    :param bilinear_interpolation: 是否使用双线性插值
    :return:
    '''
    height, width, channels = img.shape
    hRange = Interval(0, height-1)
    wRange = Interval(0, width-1)
    degree = degree*np.pi/180
    # 计算旋转后图形的高 宽
    new_height = int(height*np.cos(degree)+width*np.sin(degree))
    new_width = int(height*np.sin(degree)+width*np.cos(degree))
    new_img = np.zeros(shape=(new_height, new_width, channels), dtype=np.uint8)
    if bilinear_interpolation:
        for i in range(new_height):
            for j in range(new_width):
                new_x, new_y = j-(new_width/2), i-(new_height/2)
                old_x = (np.cos(degree)*new_x+np.sin(degree)*new_y)+(width/2)
                old_y = (-np.sin(degree)*new_x+np.cos(degree)*new_y)+(height/2)
                color = BilinearInterpolation(img, old_x, old_y, hRange, wRange)
                new_img[i, j, :] = color
    else:
        for i in range(new_height):
            for j in range(new_width):
                new_x, new_y = j-(new_width/2), i-(new_height/2)
                old_x = int(np.cos(degree)*new_x+np.sin(degree)*new_y)+int(width/2)
                old_y = int(-np.sin(degree)*new_x+np.cos(degree)*new_y)+int(height/2)
                if (old_x in wRange) and (old_y in hRange):
                    new_img[i, j, :] = img[old_y, old_x, :]
    return new_img


fig, ax = plt.subplots(1, 3, figsize=(10, 3), num='Rotate and BilinearInterpolation')
img = cv2.imread('./data/IMG1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for_img = ForRotate(img, 30)
back_img = BackRotate(img, 30, bilinear_interpolation=False)
bili_img = BackRotate(img, 30, bilinear_interpolation=True)
plt.subplot(131)
plt.title('for img')
plt.imshow(for_img)
plt.subplot(132)
plt.title('back img')
plt.imshow(back_img)
plt.subplot(133)
plt.title('bili img')
plt.imshow(bili_img)
plt.show()