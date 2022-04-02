'''
reference:https://www.youtube.com/watch?v=UcTJDamstdk&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=25
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ==============method 1: multi-resolution blending with a laplacian pyramid
def GaussionPyramid(img, level):
    '''
    形成图像的高斯金字塔
    :param img:
    :param level:
    :return:
    '''
    pyramid_images = list()
    temp = img.copy()
    for i in range(level + 1):
        dst = cv2.pyrDown(temp)
        pyramid_images.append(dst)
        temp = dst.copy()
    return pyramid_images


def LaplacianPyramid(img, level):
    '''
    形成图像的Laplacian金字塔
    :param img:
    :return:
    '''
    pyramid_images = GaussionPyramid(img, level)
    # 形成拉普拉斯金字塔
    lpls_list = list()
    lpls_list.append(pyramid_images[level])
    for i in range(level, 0, -1):
        if (i - 1) < 0:
            expand = cv2.pyrUp(pyramid_images[i])
            lpls = cv2.subtract(img, expand)
            print('error')
        else:
            expand = cv2.pyrUp(pyramid_images[i])
            lpls = cv2.subtract(pyramid_images[i - 1], expand)
        lpls_list.append(lpls)
    return lpls_list


def BlendingUseLap(source, target, mask):
    '''
    使用拉普拉斯金字塔进行图像融合
    :param source:
    :param target:
    :return:
    '''
    level = 5
    source_lpls = LaplacianPyramid(source, level)
    target_lpls = LaplacianPyramid(target, level)
    mask_Gaussion = GaussionPyramid(mask, level)
    mask_Gaussion = mask_Gaussion[::-1]
    LS = list()

    for ls, lt, mg in zip(source_lpls, target_lpls, mask_Gaussion):
        mg = mg / 255
        ls = mg * ls + (1 - mg) * lt
        LS.append(ls.astype(np.uint8))

    ls_ = LS[0]
    for i in range(1, level + 1):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_


# ==============method 2: Poisson image editing
def PoissonEditing(source, target, mask):
    # 注意h， w是反着的
    w, h, c = target.shape
    center = (int(h/2), int(w/2))
    result = cv2.seamlessClone(source, target, mask, center, cv2.MIXED_CLONE)
    return result


source_img = cv2.imread('./data/IMG26.jpg')
figure = plt.figure(num='image blending', figsize=(9, 7))
plt.subplot(2, 2, 1)
plt.title('source image')
plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))

target_img = cv2.imread('./data/IMG27.jpg')
plt.subplot(2, 2, 2)
plt.title('target image')
plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

mask_img = cv2.imread('./data/IMG26_mask.jpg')
result = BlendingUseLap(source_img, target_img, mask_img)
plt.subplot(2, 2, 3)
plt.title('LaplacianPyramid')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

result = PoissonEditing(source_img, target_img, mask_img)
plt.subplot(2, 2, 4)
plt.title('PoissonEditing')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

plt.show()
