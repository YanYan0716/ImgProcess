'''
https://www.youtube.com/watch?v=IcBzsP-fvPo&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=15&t=3239s
'''


import numpy as np
import matplotlib.pyplot as plt
import cv2


def binaryImg(img, threshold):
    '''
    图像二值化
    :param img: 图片的格式 [H, W, C]
    :param threshold: 二值化的阈值(0~255)
    :return: 二值化的图像
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    [H, W] = gray_img.shape
    binary_img = np.ones(shape=gray_img.shape)
    for i in range(H):
        for j in range(W):
            if gray_img[i, j] >= threshold:
                binary_img[i, j] = 0
    return binary_img


def earnKernel(ksize):
    assert ksize % 2 == 1, 'ksize should be odd '
    a = np.ones(shape=(int(ksize), int(ksize)))
    diag1 = np.triu(a, k=-int((ksize - 1) / 2))
    diag2 = np.tril(a, k=int((ksize - 1) / 2))
    diag = diag1 * diag2
    diag_ = cv2.rotate(diag, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    return diag * diag_


def erosion(img, ksize):
    kernel = earnKernel(ksize)
    (H, W) = img.shape

    kernel_sum = np.sum(kernel)
    new_img = np.zeros(shape=img.shape)
    a = int((ksize - 1) / 2)
    for i in range(0 + a + 1, H - a):
        for j in range(0 + a + 1, W - a):
            if np.sum(img[i - a - 1:i + a, j - a - 1:j + a] * kernel) == kernel_sum:
                new_img[i, j] = img[i, j]

    return new_img


def dilate(img, ksize):
    kernel = earnKernel(ksize)
    (H, W) = img.shape

    new_img = np.zeros(shape=img.shape)
    a = int((ksize - 1) / 2)
    for i in range(0 + a + 1, H - a):
        for j in range(0 + a + 1, W - a):
            if np.sum(img[i - a - 1:i + a, j - a - 1:j + a] * kernel) > 0:
                new_img[i, j] = 1
    return new_img


img = cv2.imread('./data/IMG17.jpg')


plt.figure(num='image erosion & dilate', figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.title('origin image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

bImg = binaryImg(img, threshold=200)
plt.subplot(2, 2, 2)
plt.title('binary image')
plt.imshow(bImg, cmap='gray')

erosion_img = erosion(bImg, 5)
plt.subplot(2, 2, 3)
plt.title('erosion image')
plt.imshow(erosion_img, cmap='gray')

dilate_img = dilate(bImg, 5)
plt.subplot(2, 2, 4)
plt.title('dilate image')
plt.imshow(dilate_img, cmap='gray')
plt.show()


def opening(img, ksize):
    '''
    图像的开运算
    :param img:
    :param ksize:
    :return:
    '''
    img = erosion(img, ksize)
    return dilate(img, ksize)


def closing(img, ksize):
    '''
    图像的闭运算
    :param img:
    :param ksize:
    :return:
    '''
    img = dilate(img, ksize)
    return erosion(img, ksize)


plt.figure(num='image opening & closing', figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.title('origin image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

bImg = binaryImg(img, threshold=200)
plt.subplot(2, 2, 2)
plt.title('binary image')
plt.imshow(bImg, cmap='gray')

oimg = opening(bImg, 5)
plt.subplot(2, 2, 3)
plt.title('opening')
plt.imshow(oimg, cmap='gray')

cimg = closing(bImg, 5)
plt.subplot(2, 2, 4)
plt.title('closing')
plt.imshow(cimg, cmap='gray')
plt.show()


def edgedetect(img, ksize):
    '''
    图像的边缘提取
    :param img:
    :param ksize:
    :return:
    '''
    img_ = erosion(img, ksize)
    # 异或运算 input数据类型为bool
    res = np.bitwise_xor(np.bool8(img), np.bool8(img_))
    return np.array(res, dtype=np.uint8)


plt.figure(num='some operations based on erosion& dilate', figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.title('origin image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

bImg = binaryImg(img, threshold=200)
plt.subplot(2, 2, 2)
plt.title('binary image')
plt.imshow(bImg, cmap='gray')

edgeimg = edgedetect(bImg, 5)
plt.subplot(2, 2, 3)
plt.title('edge detect')
plt.imshow(edgeimg, cmap='gray')

cimg = closing(bImg, 5)
plt.subplot(2, 2, 4)
plt.title('closing')
plt.imshow(cimg, cmap='gray')
plt.show()


'''====分水岭算法========='''

##########################!!!!有问题待解决

def dilate_generate(n, f, ksize, g):
    '''
    重建开操作
    :param n:
    :param f: 腐蚀后的二值图
    :param ksize: 核
    :param g: 腐蚀前的二值图
    :return:
    '''
    if n == 0:
        return f
    if n == 1:
        return np.min((dilate(f, ksize, ktype='ones'), g), axis=0)
    return dilate_generate(1, dilate_generate(n-1, f, ksize, g), ksize, g)


def erosion_generate(n, f, ksize, g):
    '''
    重建闭操作
    :param n:
    :param f:
    :param ksize:
    :param g:
    :return:
    '''
    if n == 0:
        return f
    if n == 1:
        return np.max((erosion(f, ksize, ktype='ones'), g), axis=0)
    return erosion_generate(1, erosion_generate(n-1, f, ksize, g), ksize, g)


def imreconstruce(img, ksize, mode='opening'):
    '''
    图像重建 reference：https://blog.csdn.net/csdn_yi_e/article/details/82987904
    :param img:
    :param ksize:
    :param mode:
    :return:
    '''
    bImg = binaryImg(img, threshold=200)

    if mode == 'opening':  # 重建开操作
        erosion_img = erosion(bImg, ksize=ksize, ktype='cross')  #腐蚀后的图片
        while True:
            new = dilate_generate(1, erosion_img, ksize, bImg)
            if (new == erosion_img).all():
                return erosion_img
            erosion_img = new

    elif mode == 'closing':  # 重建闭操作
        dilate_img = dilate(bImg, ksize=ksize)
        while True:
            new = erosion_generate(1, dilate_img, ksize, bImg)
            if (new == dilate_img).all():
                return dilate_img
            dilate_img = new
    else:
        print('something wrong')
        return 0


def watershed(img):
    pass


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# calculator = cv2.imread('./1233.jpg')
# plt.figure(num='some operations based on erosion& dilate', figsize=(7, 7))
# plt.subplot(2, 2, 1)
# plt.title('origin image')
# plt.imshow(cv2.cvtColor(calculator, cv2.COLOR_BGR2RGB))
#
# plt.subplot(2, 2, 2)
# plt.title('erosion image')
# plt.imshow(erosion(binaryImg(calculator, threshold=200), ksize=3, ktype='cross'), cmap='gray')
#
# res1 = imreconstruce(calculator, ksize=3, mode='opening')
# plt.subplot(2, 2, 3)
# plt.title('opening image')
# plt.imshow(res1, cmap='gray')
#
# kernel = np.ones((3, 3), np.uint8)
# sure_bg = cv2.dilate(res1, kernel, iterations=3)
# dist_transform = cv2.distanceTransform(np.array(res1*255, dtype=np.uint8), 1, 5)
# ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# sure_fg = np.uint8(sure_fg)
# sure_bg = np.uint8(sure_bg)
# unknown = cv2.subtract(sure_bg, sure_fg)
# ret, markers1 = cv2.connectedComponents(sure_fg)
#
# markers = markers1 + 1
# markers[unknown == 255] = 0
# markers3 = cv2.watershed(calculator, markers)
# calculator[markers3 == -1] = [0, 0, 255]
# plt.subplot(2, 2, 4)
# plt.title('watershed')
# plt.imshow(cv2.cvtColor(calculator, cv2.COLOR_BGR2RGB))
# # res1 = imreconstruce(calculator, ksize=3, mode='closing')
# # plt.subplot(2, 2, 4)
# # plt.title('closing image')
# # plt.imshow(res1, cmap='gray')
# plt.show()

src = cv2.imread('./1233.jpg')
img = src.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, 1, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers1 = cv2.connectedComponents(sure_fg)
markers = markers1 + 1
markers[unknown == 255] = 0

markers3 = cv2.watershed(img, markers)
img[markers3 == -1] = [0, 0, 255]

plt.subplot(241)
plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')
plt.subplot(242)
plt.imshow(thresh, cmap='gray')
plt.title('Threshold')
plt.axis('off')
plt.subplot(243)
plt.imshow(sure_bg, cmap='gray')
plt.title('Dilate')
plt.axis('off')
plt.subplot(244)
plt.imshow(dist_transform, cmap='gray')
plt.title('Dist Transform')
plt.axis('off')
plt.subplot(245)
plt.imshow(sure_fg, cmap='gray')
plt.title('Threshold')
plt.axis('off')
plt.subplot(246)
plt.imshow(unknown, cmap='gray')
plt.title('Unknow')
plt.axis('off')
plt.subplot(247)
plt.imshow(np.abs(markers), cmap='jet')
plt.title('Markers')
plt.axis('off')
plt.subplot(248)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Result')
plt.axis('off')

plt.show()
