'''
reference:https://www.youtube.com/watch?v=NbQY1x8H6QQ&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=7&t=917s
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def basicfunction(m, n):
    '''
    傅里叶变换的基本函数
    :param m:
    :param n:
    :return: 0~1的范围内
    '''
    N = 256
    x, y = np.arange(N), np.arange(N)
    x, y = np.meshgrid(x, y)
    # 上述两行代码的另一种写法
    # x = np.int8([]).reshape(0, 255)
    # tmp_x = np.array((np.arange(0, N - 1))).reshape(1, 255)
    # for i in range(255):
    #     x = np.append(x, tmp_x, axis=0)
    # y = np.transpose(x)

    complex_num = 0+1j
    im = np.real(np.exp(-complex_num*2 * np.pi * (m * x / N + n * y / N)))
    if m == 0 and n == 0:
        im = np.round(im)
    return im


fig, ax = plt.subplots(1, 3, figsize=(10, 5), num='median')
plt.subplot(131)
plt.title('basic function(0, 0)')
plt.imshow(Image.fromarray(255*basicfunction(0, 0)))

plt.subplot(132)
plt.title('basic function(1, 0)')
plt.imshow(Image.fromarray(255*basicfunction(1, 0)))

plt.subplot(133)
plt.title('basic function(2, 0)')
im = basicfunction(2, 0)
plt.imshow(Image.fromarray(255*im))
plt.show()


# 图片显示傅里叶变换
N = 256
x, y = np.arange(N), np.arange(N)
x, y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, im, cmap='rainbow', rstride=2, cstride=2)
plt.title('surface about basic_function(2, 0)')
plt.show()


fig, ax = plt.subplots(1, 3, figsize=(10, 5), num='median')
plt.subplot(131)
plt.title('basic function(1, 0)')
plt.imshow(Image.fromarray(255*basicfunction(1, 0)))

plt.subplot(132)
plt.title('basic function(0, 1)')
plt.imshow(Image.fromarray(255*basicfunction(0, 1)))

plt.subplot(133)
plt.title('basic function(1, 1)')
im = basicfunction(1, 1)
plt.imshow(Image.fromarray(255*im))
plt.show()


# 图片显示傅里叶变换
N = 256
x, y = np.arange(N), np.arange(N)
x, y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.gca(projection='3d')
im = basicfunction(2, 3)
surf = ax.plot_surface(x, y, im, cmap='rainbow', rstride=2, cstride=2)
plt.title('surface about basic_function(2, 0)')
plt.show()


# 图像的傅里叶变换
fig, ax = plt.subplots(1, 3, figsize=(10, 3), num='image fft')
img = cv2.imread('./data/IMG1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(131)
plt.title('original img')
plt.imshow(img, cmap='gray')

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
plt.subplot(132)
plt.title('fftshift about magnitude spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')

img_shift = np.fft.fftshift(img)
plt.subplot(133)
plt.title('fftshift about img')
plt.imshow(img_shift, cmap='gray')
plt.show()


fig, ax = plt.subplots(3, 3, figsize=(10, 10), num='fft with different image')
img = cv2.imread('./data/IMG2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(331)
plt.title('original img')
plt.imshow(img, cmap='gray')

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
plt.subplot(332)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('图的边缘拼接在一起差别不大')
plt.imshow(magnitude_spectrum, cmap='gray')

img_shift = np.fft.fftshift(img)
plt.subplot(333)
plt.title('fftshift about img')
plt.imshow(img_shift, cmap='gray')

img = cv2.imread('./data/IMG4.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(334)
plt.title('original img')
plt.imshow(img, cmap='gray')

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
plt.subplot(335)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('图中有很多斜线')
plt.imshow(magnitude_spectrum, cmap='gray')

img_shift = np.fft.fftshift(img)
plt.subplot(336)
plt.title('fftshift about img')
plt.imshow(img_shift, cmap='gray')

img = cv2.imread('./data/IMG5.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(337)
plt.title('original img')
plt.imshow(img, cmap='gray')

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
plt.subplot(338)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('图中有很对边缘即高频较多')
plt.imshow(magnitude_spectrum, cmap='gray')

img_shift = np.fft.fftshift(img)
plt.subplot(339)
plt.title('fftshift about img')
plt.imshow(img_shift, cmap='gray')
plt.show()


fig, ax = plt.subplots(3, 2, figsize=(10, 10), num='fft with different image2')
img = cv2.imread('./data/IMG6.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(231)
plt.title('竖条纹')
plt.imshow(img, cmap='gray')

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
plt.subplot(234)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('竖条纹 magnitude_spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')


img = cv2.imread('./data/IMG7.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(232)
plt.title('original img')
plt.imshow(img, cmap='gray')

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
plt.subplot(235)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('横条纹 magnitude_spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')


img = cv2.imread('./data/IMG8.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(233)
plt.title('original img')
plt.imshow(img, cmap='gray')

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
plt.subplot(236)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('斜条纹 magnitude_spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.show()