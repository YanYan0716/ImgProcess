'''
空间滤波
reference:https://www.youtube.com/watch?v=q9AqlQ274ss&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=6
'''
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

'''image smooth==========================='''


def SmoothFilter(image, kernel_size):
    H, W, C = image.shape
    smooth_filter = np.ones(shape=(kernel_size, kernel_size, C)) / np.power(kernel_size, 2)
    new_img = np.zeros(shape=(H, W, C))
    radius = int((kernel_size - 1) / 2)

    for h in range(H):
        for w in range(W):
            left = np.clip(h - radius, 1, H - (kernel_size - radius) - 1)
            top = np.clip(w - radius, 1, W - (kernel_size - radius) - 1)
            new_img[h, w, :] = np.einsum('ijp,ijp->p', image[left:left + kernel_size, top:top + kernel_size],
                                         smooth_filter)

    new_img = np.array(new_img, dtype=np.uint8)
    return new_img


fig, ax = plt.subplots(1, 3, figsize=(10, 10), num='smooth filter')
img = cv2.cvtColor(cv2.imread('./data/IMG1.jpg'), cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.title('origin')
plt.imshow(img)

img_ = cv2.blur(img, (3, 3))
plt.subplot(222)
plt.title('smooth by opencv')
plt.imshow(img_)

KERNELSIZE = 3
kernel = np.ones(shape=(KERNELSIZE, KERNELSIZE), dtype=np.float32) / np.power(KERNELSIZE, 2)
(r, g, b) = cv2.split(img)
r = cv2.filter2D(r, -1, kernel=kernel)
g = cv2.filter2D(g, -1, kernel=kernel)
b = cv2.filter2D(b, -1, kernel=kernel)
img_2d = cv2.merge((r, g, b))
plt.subplot(222)
plt.title('filter by opencv')
plt.imshow(img_2d)

img_self = SmoothFilter(img, kernel_size=3)
plt.subplot(223)
plt.title('smooth from myself')
plt.imshow(img_self)
plt.show()


def Filter2ThreeD(image, kernel):
    H, W, C = image.shape
    new_img = np.zeros(shape=(H, W, C))
    kernel_size = kernel.shape[0]
    radius = int((kernel.shape[0] - 1) / 2)

    for h in range(H):
        for w in range(W):
            left = np.clip(h - radius, 1, H - (kernel_size - radius) - 1)
            top = np.clip(w - radius, 1, W - (kernel_size - radius) - 1)
            new_img[h, w, :] = np.einsum('ijp,ij->p', image[left:left + kernel_size, top:top + kernel_size], kernel)
    new_img = np.array(np.clip(new_img, 0, 255), dtype=np.uint8)
    return new_img


'''image shaperen==========================='''
fig, ax = plt.subplots(1, 3, figsize=(10, 10), num='shaperen filter')
img = cv2.cvtColor(cv2.imread('./data/IMG1.jpg'), cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.title('origin')
plt.imshow(img)

KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
(r, g, b) = cv2.split(img)
r = cv2.filter2D(r, -1, kernel=KERNEL)
g = cv2.filter2D(g, -1, kernel=KERNEL)
b = cv2.filter2D(b, -1, kernel=KERNEL)
img_ = cv2.merge((r, g, b))
plt.subplot(222)
plt.title('shaperen filter by opencv')
plt.imshow(img_)

img_self = Filter2ThreeD(img, kernel=KERNEL)
plt.subplot(223)
plt.title('shaperen from myself')
plt.imshow(img_self)
plt.show()

'''sobel detector==========================='''


def Filter2TwoD(image, kernel):
    H, W = image.shape
    new_img = np.zeros(shape=(H, W))
    kernel_size = kernel.shape[0]
    radius = int((kernel.shape[0] - 1) / 2)

    for h in range(H):
        for w in range(W):
            left = np.clip(h - radius, 1, H - (kernel_size - radius) - 1)
            top = np.clip(w - radius, 1, W - (kernel_size - radius) - 1)
            new_img[h, w] = np.sum(image[left:left + kernel_size, top:top + kernel_size]*kernel)
    new_img = np.array(np.clip(new_img, 0, 255), dtype=np.uint8)
    return new_img


fig, ax = plt.subplots(1, 3, figsize=(6, 10), num='sobel edge detector')
img = cv2.cvtColor(cv2.imread('./data/IMG1.jpg'), cv2.COLOR_BGR2RGB)
plt.subplot(321)
plt.title('origin')
plt.imshow(img)


# sobel horizontal detector
SHD = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
# sobel vertical detector
SVD = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

sobelx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
_, sobelx = cv2.threshold(sobelx, 127, 255, cv2.THRESH_BINARY)
plt.subplot(323)
plt.title('sobel horizontal by opencv')
plt.imshow(sobelx, cmap='gray')

imgx_self = Filter2TwoD(img, kernel=SHD)
plt.subplot(324)
plt.title('sobel horizontal by myself')
plt.imshow(imgx_self, cmap='gray')

sobely = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
_, sobely = cv2.threshold(sobely, 127, 255, cv2.THRESH_BINARY)
plt.subplot(325)
plt.title('sobel vertical by opencv')
plt.imshow(sobely, cmap='gray')

imgy_self = Filter2TwoD(img, kernel=SVD)
plt.subplot(326)
plt.title('sobel vertical by myself')
plt.imshow(imgy_self, cmap='gray')
plt.show()


'''MedianFilter for pepper and salt noise==========================='''


def MedianFilter(image, kernel_size):
    H, W = image.shape
    new_img = np.zeros(shape=(H, W))
    radius = int((kernel_size - 1) / 2)

    for h in range(H):
        for w in range(W):
            left = np.clip(h - radius, 1, H - (kernel_size - radius) - 1)
            top = np.clip(w - radius, 1, W - (kernel_size - radius) - 1)
            new_img[h, w] = int(np.median(image[left:left+kernel_size, top:top+kernel_size]))
    new_img = np.array(np.clip(new_img, 0, 255), dtype=np.uint8)
    return new_img


fig, ax = plt.subplots(1, 3, figsize=(10, 5), num='median')
img = cv2.cvtColor(cv2.imread('./data/IMG3.png'), cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.title('papper and salt')
plt.imshow(img)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_ = MedianFilter(img, kernel_size=3)
plt.subplot(122)
plt.title('papper and salt with median filter')
plt.imshow(Image.fromarray(img_), cmap='gray')
plt.show()