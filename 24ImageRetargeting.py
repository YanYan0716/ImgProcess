'''
reference:https://www.youtube.com/watch?v=w8pjvtnjRPs&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=27
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import random


# ==================seam carving===============================================
def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    filter_dv = np.stack([filter_dv] * 3, axis=2)
    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))
    energy_map = convolved.sum(axis=2)
    return energy_map


def minimum_seam(img, energy_img):
    r, c, _ = img.shape
    backtrack = np.zeros_like(energy_img, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(energy_img[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = energy_img[i - 1, idx + j]
            else:
                idx = np.argmin(energy_img[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = energy_img[i - 1, idx + j - 1]
            energy_img[i, j] += min_energy
    return backtrack


def carve_column(img):
    energy_img = calc_energy(img)
    r, c, _ = img.shape
    backtrack = minimum_seam(img, energy_img)
    mask = np.ones((r, c), dtype=np.bool)
    j = np.argmin(energy_img[-1])

    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask]*3, axis=2)
    img = img[mask].reshape((r, c-1, 3))
    return img


def SeamCarving(img, scale_c):
    '''
    reference:https://zhuanlan.zhihu.com/p/38974520
    :param img:
    :param scale_c:
    :return:
    '''
    energy_img = calc_energy(img)
    energy_img = (energy_img - np.min(energy_img)) / (np.max(energy_img) - np.min(energy_img))

    r, c, _ = img.shape
    new_c = int(scale_c*c)

    for i in range(c-new_c):
        img = carve_column(img)
    return (energy_img * 255).astype(np.uint8), img


# img = cv2.imread('./data/IMG30.jpg')
# figure = plt.figure(num='image blending', figsize=(9, 4))
# plt.subplot(1, 3, 1)
# plt.title('source image')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
# enerey_img, result = SeamCarving(img, 0.7)
# plt.subplot(1, 3, 2)
# plt.title('seam carving enerey_img')
# plt.imshow(enerey_img, cmap='gray')
#
# plt.subplot(1, 3, 3)
# plt.title('seam carving image')
# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
# plt.show()


# ==================PatchMatch===============================================
def InitOff(mask):
    r, c = mask.shape
    Off = np.zeros(shape=(r, c, 2), dtype=np.float32)

    for i in range(r):
        for j in range(c):
            if mask[i][j] == 0:
                Off[i, j, 0] = 0
                Off[i, j, 1] = 0
            else:
                r_col = int(random.random()*c)
                r_row = int(random.random()*r)
                r_col = r_col if (r_col+j) < c else c-r_col
                r_row = r_row if (r_row+i) < r else r-r_row


def PatchMatch(img, mask):
    '''
    reference:https://www.jb51.net/article/210277.htm
    :param img:
    :param mask:
    :return:
    '''

    return img


img = cv2.imread('./data/IMG25.jpg')
mask = cv2.imread('./data/IMG25_mask.jpg')
figure = plt.figure(num='image inpainting', figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.title('source image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

result = PatchMatch(img, mask)
plt.subplot(1, 2, 2)
plt.title('PatchMatch')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

plt.show()