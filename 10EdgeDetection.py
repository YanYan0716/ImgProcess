'''
reference:https://www.youtube.com/watch?v=APBXfqVccS0&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=10
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(3, 3, figsize=(10, 10), num='gradient')
im = cv2.imread('./data/IMG10.jpg')
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
plt.subplot(331)
plt.title('original image')
plt.imshow(gray_im, cmap='gray')

hx = np.array([[-1, 1]])
hy = np.transpose(np.array([[-1, 1]]))

gx = cv2.filter2D(gray_im.astype(np.float), ddepth=-1, kernel=hx)
plt.subplot(332)
plt.title('gradient in x')
plt.imshow(np.abs(gx), cmap='gray')

gy = cv2.filter2D(gray_im.astype(np.float), ddepth=-1, kernel=hy)
plt.subplot(333)
plt.title('gradient in y')
plt.imshow(np.abs(gy), cmap='gray')

plt.subplot(334)
plt.title('colormap with x')
plt.imshow(np.abs(gx), cmap=plt.get_cmap('rainbow'))

ret, th = cv2.threshold(np.abs(gx), 20, 255, cv2.THRESH_BINARY)
plt.subplot(335)
plt.title('threshold with x')
plt.imshow(th, cmap='gray')

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
gx_sobel = cv2.filter2D(gray_im, ddepth=-1, kernel=sobel_x)
plt.subplot(336)
plt.title('sobel with x')
plt.imshow(np.abs(gx_sobel), cmap=plt.get_cmap('rainbow'))

m = np.sqrt(np.power(gx.astype(np.float), 2)+np.power(gy.astype(np.float), 2))
plt.subplot(337)
plt.title('magnitude about image')
plt.imshow(m.astype(np.uint), cmap='gray')

alpha = np.arctan2(gy.astype(np.float), gx.astype(np.float))*180/np.pi
plt.subplot(338)
plt.title('alpha in image')
plt.imshow(alpha, cmap=plt.get_cmap('rainbow'))

e = (np.abs((alpha-30) > 10)) & (m > 20)
plt.subplot(339)
plt.title('condition in image')
plt.imshow(e, cmap='gray')

plt.show()


plt.subplot()
plt.title('alpha in image with more details')
plt.imshow(alpha, cmap=plt.get_cmap('rainbow'))
plt.colorbar()
plt.show()


'''===============laplacian of gaussian(LOG)========='''


def createLOGKernel(sigma, size):
    H, W = size
    r, c = np.mgrid[0:H:1.0, 0:W:1.0]
    r -= (H-1)/2
    c -= (W-1)/2
    sigma2 = np.power(sigma, 2.)
    norm2 = np.power(r, 2.) + np.power(c, 2.0)
    LogKernel = (norm2/sigma2 -2)*np.exp(-norm2/(2*sigma2))
    return LogKernel


H, W = [101, 101]
h = createLOGKernel(10, [101, 101])
fig = plt.figure()
ax = fig.gca(projection='3d')
r, c = np.mgrid[0:H:1.0, 0:W:1.0]
ax.plot_surface(r, c, -h, cmap=plt.get_cmap('rainbow'), rstride=2, cstride=2)
plt.title("laplacian of gaussian's filter")
plt.show()


fig, ax = plt.subplots(1, 3, figsize=(10, 3), num='LOG')
im = cv2.imread('./data/IMG10.jpg')
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
plt.subplot(131)
plt.title('original image')
plt.imshow(gray_im, cmap='gray')

LOG_10 = cv2.filter2D(gray_im.astype(np.float), ddepth=-1, kernel=h)
plt.subplot(132)
plt.title('LOG with sigma=10')
plt.imshow(LOG_10, cmap=plt.get_cmap('rainbow'))
plt.colorbar()

h = createLOGKernel(3, [101, 101])
LOG_3 = cv2.filter2D(gray_im.astype(np.float), ddepth=-1, kernel=h)
plt.subplot(133)
plt.title('LOG with sigma=3')
plt.imshow(LOG_3, cmap=plt.get_cmap('rainbow'))
plt.colorbar()

plt.show()


'''===============Difference of gaussian(DOG)========='''


def gauss_filter(kernel_size, sigma=1):
    kx = cv2.getGaussianKernel(kernel_size[0], sigma)
    ky = cv2.getGaussianKernel(kernel_size[1], sigma)
    return np.multiply(kx, ky.T)

h1 = gauss_filter([101, 101], 20)
h2 = gauss_filter([101, 101], 10)

H, W = [101, 101]
r, c = np.mgrid[0:H:1.0, 0:W:1.0]
fig = plt.figure('DOG', figsize=(10, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(r, c, h1, cmap=plt.get_cmap('rainbow'), rstride=2, cstride=2)
plt.title("gaussian's filter sigma=20")

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(r, c, h2, cmap=plt.get_cmap('rainbow'), rstride=2, cstride=2)
plt.title("gaussian's filter sigma=10")

ax2 = fig.add_subplot(133, projection='3d')
ax2.plot_surface(r, c, -(h1-h2), cmap=plt.get_cmap('rainbow'), rstride=2, cstride=2)
plt.title("abs with fig1 & fig2")

plt.show()


'''===============canny==================='''

fig, ax = plt.subplots(1, 3, figsize=(7, 3), num='canny')
im = cv2.imread('./data/IMG10.jpg')
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
plt.subplot(131)
plt.title('original image')
plt.imshow(gray_im, cmap='gray')

blur = cv2.GaussianBlur(gray_im, (5, 5), 0)
plt.subplot(132)
plt.title('blur')
plt.imshow(blur, cmap='gray')

canny = cv2.Canny(blur, 50, 150)
plt.subplot(133)
plt.title('canny')
plt.imshow(canny, cmap='gray')

plt.show()

