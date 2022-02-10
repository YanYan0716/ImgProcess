'''
reference:
https://blog.csdn.net/zddblog/article/details/7521424
https://blog.csdn.net/hit2015spring/article/details/52972890
https://blog.csdn.net/hit2015spring/article/details/52895367?spm=1001.2014.3001.5502
https://wenku.baidu.com/view/ebbf1540ad51f01dc381f176.html
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


SIFT_FIXPT_SCALE = 1
def GenerateGaussianFilter(sigma):
    '''
    生成标准差为sigma的高斯核，高斯核的大小遵循 6*sigma 原则
    :param sigma:
    :return: 生成的高斯核 大小为[1, size]
    '''
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    radius = int((size - 1) / 2)
    kernel = np.zeros(shape=[size])
    a = 2 * sigma * sigma
    for i in range(size):
        kernel[i] = np.exp(-np.power(-radius + i, 2) / a) / np.sqrt(a * np.pi)
    return kernel, radius


def GaussianBlur(img, sigma):
    '''
    对图片进行高斯模糊 利用离散二维高斯函数  先进行宽的模糊再进行高的模糊  仅限于灰度图
    :param img:
    :param sigma: 高斯函数的标准差 值越大图像越模糊
    :return: 模糊后的图像
    '''
    kernel, radius = GenerateGaussianFilter(sigma)
    [H, W] = img.shape
    new_img = img.copy()

    # 宽方向的高斯模糊
    for h in range(0, H):
        for w in range(radius, W - radius):
            left = np.clip(w - radius, 0, W)
            right = np.clip(w + radius + 1, 0, W)
            new_img[h, w] = np.clip(np.sum(kernel * img[h, left:right]), 0, 255)

    # 高方向的高斯模糊
    img = new_img.copy()
    for w in range(0, W):
        for h in range(radius, H - radius):
            left = np.clip(h - radius, 0, H)
            right = np.clip(h + radius + 1, 0, H)
            new_img[h, w] = np.clip(np.sum(kernel * img[left:right, w]), 0, 255)

    return new_img


def downsample(img):
    img = img[::2, ::2]
    return img


def earn_sigma(sigma0, S):
    k = np.power(2, 1/3)
    sigma_list = list()
    for i in range(0, S):
        sigma_list.append(np.sqrt(np.power(np.power(k, i)*sigma0, 2)-np.power(np.power(k, i-1)*sigma0, 2)))
    return sigma_list


def diff1(x1, x2, h):
    '''
    图像的一阶求导函数
    :param x1:
    :param x2:
    :return:
    '''
    return (x2 - x1) / (2. * h)


def diff2(x1, x2, x3, h):
    '''
    图像的二阶求导函数
    :param x1:
    :param x2:
    :param x3:
    :param h:
    :return:
    '''
    return (x1 + x3 - 2. * x2) / float(np.power(h, 2))


def diff2_(x1, x2, x3, x4, h):
    '''
    图像的二阶求偏导
    :param x1:
    :param x2:
    :param x3:
    :param x4:
    :param h:
    :return:
    '''
    return ((x3 + x1) - (x2 + x4)) / (4. * h * h)


def adjustLocalExtrema(DOG, octv, layer, loc_h, loc_w, nOctaveLayers, contrastThreshold, edgeThreshold, h):
    '''
    reference:https://blog.csdn.net/hit2015spring/article/details/52972890
    :param DOG: DOG金字塔
    :param octv: 极值点所在的组数
    :param layer: 极值点所在的层数
    :param loc_h: 极值点的H坐标
    :param loc_w: 极值点的W坐标
    :param nOctaveLayers: 本组一共有多少层
    :param contrastThreshold:
    :param edgeThreshold:
    :param h:
    :return:
    '''
    # print(octv, layer)
    SIFT_MAX_INTTERP_STEPS = 5
    INT_MAX = np.Inf
    SIFT_FIXPT_SCALE = 1.
    # h = 1 / (255 * SIFT_FIXPT_SCALE)

    before = DOG[octv][layer - 1]
    target = DOG[octv][layer]
    [H, W] = target.shape
    after = DOG[octv][layer + 1]
    x, y = loc_w, loc_h
    xc, xr, xi = 0, 0, 0
    # 第二轮筛选
    for i in range(SIFT_MAX_INTTERP_STEPS):
        diff_x = diff1(target[y, x - 1], target[y, x + 1], h)
        diff_y = diff1(target[y - 1, x], target[y + 1, x], h)
        diff_sigma = diff1(before[y, x], after[y, x], h)
        diff_f1 = np.array([diff_x, diff_y, diff_sigma])

        diff_x2 = diff2(target[y, x - 1], target[y, x], target[y, x + 1], h)
        diff_y2 = diff2(target[y - 1, x], target[y, x], target[y + 1, x], h)
        diff_sigma2 = diff2(before[y, x], target[y, x], after[y, x], h)

        diff_xy = diff2_(target[y - 1, x - 1], target[y - 1, x + 1], target[y + 1, x + 1], target[y + 1, x - 1], h)
        diff_xsigma = diff2_(before[y, x - 1], before[y, x + 1], after[y, x + 1], after[y, x - 1], h)
        diff_ysigma = diff2_(before[y - 1, x], before[y + 1, x], after[y + 1, x], after[y - 1, x], h)
        diff_f2 = np.array([
            [diff_x2, diff_xy, diff_xsigma],
            [diff_xy, diff_y2, diff_ysigma],
            [diff_xsigma, diff_ysigma, diff_sigma2]
        ])
        try:
            offset = np.linalg.solve(diff_f2, diff_f1)
        except:
            return False
        # x轴 y轴 层 的偏移量
        [xc, xr, xi] = -offset

        if abs(xc) < 0.5 and abs(xr) < 0.5 and abs(xi) < 0.5:
            break
        if abs(xi) > INT_MAX or abs(xr) > INT_MAX or abs(xc) > INT_MAX:
            return False
        x += int(np.round(xc))
        y += int(np.round(xr))
        layer += int(np.round(xi))

        # 如果超出金字塔的坐标范围 也说明不是极值点 也要删除
        if layer < 1 or layer > nOctaveLayers or x < 3 or x > W - 2 or y < 3 or y > H - 2:
            return False

    # 第三轮筛选
    value1 = diff1(target[y, x - 1], target[y, x + 1], h)
    value2 = diff1(target[y - 1, x], target[y + 1, x], h)
    value3 = diff1(before[y, x], after[y, x], h)
    t = np.dot(np.array([value1, value2, value3]), np.array([xc, xr, xi]))
    contr = target[y, x] + t * 0.5
    if abs(contr) * nOctaveLayers < contrastThreshold:
        return False

    # 第四轮筛选
    dxx = diff2(target[y, x - 1], target[y, x], target[y, x + 1], h)
    dyy = diff2(target[y - 1, x], target[y, x], target[y + 1, x], h)
    dxy = diff2_(target[y - 1, x - 1], target[y + 1, x - 1], target[y + 1, x + 1], target[y - 1, x + 1], h)
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy

    if det <= 0 or tr * tr * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * det:
        return False

    return x+int(np.round(xc)), y + int(np.round(xr))


def GaussianPyramid(img, n=3, s=3, sigma=1.6):
    img = np.array(img, dtype=np.float32)
    GP = list()  # 高斯金字塔
    GP_interval_num = s + 3
    DOG = list()  # 高斯差分金字塔(DOG)
    DOG_interval_num = s + 2

    sigma_list = earn_sigma(sigma, GP_interval_num)

    # 多尺度空间的高斯金字塔的构建
    for i in range(n):
        octave = list()
        for single_sigma in sigma_list:
            blur = GaussianBlur(img, single_sigma)
            octave.append(blur)
        octave = np.array(octave, dtype=np.float32)
        GP.append(octave)
        img = downsample(octave[-3])

    # DOG的构建
    for i in range(n):
        octave = list()
        for le in range(DOG_interval_num):
            diff = np.clip(GP[i][le+1] - GP[i][le], 0, 255)
            octave.append(diff)
        octave = np.array(octave)
        DOG.append(octave)

    # 空间极值点检测 -> 1. 检索邻域内的极值点并存储
    extreme = list()  # 存储极值点 极大值为1 极小值为-1 其余为0
    extreme_position = list()
    for i in range(n):
        octave = list()
        octave_position = list()
        for j in range(s):
            before = DOG[i][j]
            target = DOG[i][j + 1]
            after = DOG[i][j + 2]
            patten = np.zeros(shape=target.shape)
            [sH, sW] = target.shape
            for sh in range(1, sH - 1):
                for sw in range(1, sW - 1):
                    all_neighbor = [before[sh - 1:sh + 2, sw - 1:sw + 2], target[sh - 1:sh + 2, sw - 1:sw + 2],
                                    after[sh - 1:sh + 2, sw - 1:sw + 2]]
                    min_neighbor = np.min(all_neighbor)
                    max_neighbor = np.max(all_neighbor)
                    result = adjustLocalExtrema(DOG, i, j+1, sh, sw, 5, 0.04, 10, 1.)
                    if target[sh, sw] == min_neighbor and result != False:
                        patten[sh, sw] = -1
                        # octave_position.append([sh, sw])
                    elif target[sh, sw] == max_neighbor and result != False:
                        patten[sh, sw] = 1
                        octave_position.append([sh, sw])
                    else:
                        pass
            octave.append(patten)
            extreme_position.append(np.array(octave_position))
        octave = np.array(octave)
    return GP, DOG, extreme_position


figure = plt.figure(num='sift algorithm-1', figsize=(7, 4))
img = cv2.imread('./apple.jpg')
plt.subplot(1, 3, 1)
plt.xticks([])
plt.yticks([])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(1, 3, 2)
plt.xticks([])
plt.yticks([])
plt.imshow(gray_img, cmap='gray')

blur_img = GaussianBlur(gray_img, 1.6)
plt.subplot(1, 3, 3)
plt.xticks([])
plt.yticks([])
plt.imshow(blur_img, cmap='gray')
plt.show()

figure = plt.figure(num='sift algorithm-2', figsize=(10, 7))
GP, DOG, extreme = GaussianPyramid(blur_img)
print(len(extreme))
print(extreme[0].shape)
index = 0
for i in range(3):
    for j in range(GP[0].shape[0]):
        index += 1
        plt.subplot(3, 6, index)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(GP[i][j], cmap='gray')
plt.show()

figure = plt.figure(num='sift algorithm-3', figsize=(10, 7))
index = 0
for i in range(3):
    for j in range(DOG[0].shape[0]):
        index += 1
        plt.subplot(3, 5, index)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(DOG[i][j], cmap='gray')
plt.show()
#
# figure = plt.figure(num='sift algorithm-4', figsize=(10, 7))
# for j in range(3):
#     plt.subplot(1, 3, j+1)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.xticks([])
#     plt.yticks([])
#     plt.scatter(extreme[j][:, 1], extreme[j][:, 0], marker='+', color='y')
#
# plt.show()


# sift in opencv
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = './apple.jpg'

sift = cv2.SIFT_create()
img1 = cv2.imread(imgname1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述

img3 = cv2.drawKeypoints(img1, kp1, img1, color=(0, 255, 0))  # 画出特征点，并显示为红色圆圈
cv2.imshow("point", img3)  # 拼接显示为gray
cv2.waitKey(0)
cv2.destroyAllWindows()



'''


import cv2
import numpy as np
import matplotlib.pyplot as plt


def Hist(img):
    histimg = cv2.calcHist([img], [1], None, [256], [0, 256])
    cv2.normalize(histimg, histimg, 0, 255*0.9, cv2.NORM_MINMAX)



def my_sift(hd, photo):
    MIN_MATCH_COUNT = 25
    template = cv2.imread(hd, 0)
    height, width = template.shape[0], template.shape[1]
    max_length = max(height, width)
    if max_length > 1000:
        ratio = 1000 / float(max_length)
        template = cv2.resize(template, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)

    target = cv2.imread(photo, 0)
    height, width = target.shape[0], target.shape[1]
    max_length = max(height, width)
    if max_length > 1920:
        ratio = 1920 / float(max_length)
        target = cv2.resize(target, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    target_height, target_width = target.shape[0], target.shape[1]
    target_area = target_height * target_width

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)
    print('---------------')
    print(len(kp1), len(kp2))

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    print(len(good))
    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        cv2.polylines(target, [np.int32(dst)], True, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        print('Not enough matches are found - %d%d' % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=2
    )

    img = cv2.imread(photo)
    result = cv2.drawMatches(template, kp1, img, kp2, good, None, **draw_params)

    dst = dst.reshape(4, 2)

    x = [dst[i][0] for i in range(len(dst))]
    x_min = int(np.min(x))
    x_max = int(np.max(x))
    y = [dst[i][1] for i in range(len(dst))]
    y_min = int(np.min(y))
    y_max = int(np.max(y))

    img = cv2.imread(photo)
    dst = dst.reshape(4, 1, 2)
    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
    roi_width = x_max - x_min
    roi_height = y_max - y_min
    print(roi_width, roi_height, target_width, target_height)
    print(roi_width*roi_height, target_area)
    if ((target_width/5) < roi_width < (target_width) or \
            (target_height/5) < roi_height < (target_height)) and \
            (target_area/10) < (roi_width*roi_height) < (target_area*2/3):
        print('yes')
        # cv2.polylines(img, [np.int32(dst)], isClosed=True, color=(155, 244, 123), thickness=3)
        # cv2.imshow('as', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.imshow(result, 'gray')
        # plt.show()
    else:
        # cv2.polylines(img, [np.int32(dst)], isClosed=True, color=(155, 244, 123), thickness=3)
        # cv2.imshow('as', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        plt.imshow(result, 'gray')
        plt.show()


if __name__ == '__main__':
    # my_sift('F:\\PROJECT\\ComDis\\183372.jpg', 'F:\\PROJECT\\ComDis\\123.png')
    path1 = 'HD.jpg'
    path2 = 'SCREEN.jpg'
    my_sift(path1, path1)

# from imutils.perspective import four_point_transform
# import imutils


#
# a = cv2.imread(path1)
# dst = dst.reshape(4, 2)
# dst[:, 0] = np.clip(dst[:, 0], 0, a.shape[0])
# dst[:, 1] = np.clip(dst[:, 1], 0, a.shape[1])
#
# transimg = cv2.imread(path2)
# transimg = four_point_transform(transimg, np.int32(dst).reshape(4, 2))
# transimg = cv2.resize(transimg, a.shape[:2])
# cv2.imshow('res', transimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
'''
