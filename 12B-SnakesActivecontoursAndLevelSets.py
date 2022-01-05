'''
reference:
https://www.youtube.com/watch?v=RJEMDkhVgqQ&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=14
https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_morphsnakes.html
https://wenku.baidu.com/view/ce52313181c758f5f71f6767.html
https://blog.csdn.net/cfan927/article/details/108884457
https://agustinus.kristia.de/techblog/2016/11/05/levelset-method/
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from skimage.filters import sobel

'''==========snake(active contour) ========================'''


def edgedetect(img):
    edge_img = [sobel(img[:, :, 0]), sobel(img[:, :, 1]),
                sobel(img[:, :, 2])]
    return sum(edge_img)


def snake(img, snake, w_edge=0, w_line=1, convergence=0.1, alpha=0.01, beta=0.1, gamma=0.01, max_iterations=2500,
          max_px_move=1):
    '''
    参照skimage中的函数进行实现，主要是为了理解各个参数的含义
    :param img:
    :param snake:
    :param w_edge:
    :param w_line:
    :param convergence:
    :param alpha:
    :param beta:
    :param gamma:
    :param max_iterations:
    :param max_px_move:
    :return:
    '''
    snake_xy = snake[:, ::-1]
    x, y = snake_xy[:, 0].astype(np.float32), snake_xy[:, 1].astype(np.float32)
    convergence_order = 10
    n = len(x)
    xsave = np.empty((convergence_order, n))
    ysave = np.empty((convergence_order, n))

    ### 内部能量
    a = np.roll(np.eye(n), -1, axis=0) + np.roll(np.eye(n), -1, axis=1) - 2 * np.eye(n)
    b = np.roll(np.eye(n), -2, axis=0) + np.roll(np.eye(n), -2, axis=1) - \
        4 * np.roll(np.eye(n), -1, axis=0) - 4 * np.roll(np.eye(n), -1, axis=1) + 6 * np.eye(n)

    A = -alpha * a + beta * b
    inv = np.linalg.inv(A + gamma * np.eye(n))  # 矩阵求逆

    ### 外部能量 -> 图像能量 -> 边缘能量
    edge_img = edgedetect(img)

    ### 外部能量 -> 图像能量的加权
    img = w_line * np.sum(img, axis=2) + w_edge * edge_img

    intp = RectBivariateSpline(
        np.arange(img.shape[1]),
        np.arange(img.shape[0]),
        edge_img.T,
        kx=2,
        ky=2,
        s=0
    )

    plt.ion()
    plt.figure(num='snake')
    plt.clf()
    plt.imshow(cv2.cvtColor(cv2.imread('./data/IMG16.jpg'), cv2.COLOR_BGR2RGB))
    plt.pause(0.1)

    for i in range(max_iterations):
        fx = intp(x, y, dx=1, grid=False)  # 求偏导 x方向
        fy = intp(x, y, dy=1, grid=False)  # 求偏导 y方向

        xn = inv @ (gamma * x + fx)
        yn = inv @ (gamma * y + fy)

        dx = max_px_move * np.tanh(xn - x)
        dy = max_px_move * np.tanh(yn - y)

        x += dx
        y += dy

        j = i % (convergence_order + 1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave - x[None, :]) + np.abs(ysave - y[None, :]), 1))
            if dist < convergence:
                break

        plt.clf()
        plt.imshow(cv2.cvtColor(cv2.imread('./data/IMG16.jpg'), cv2.COLOR_BGR2RGB))
        plt.plot(x, y, linewidth=2, color='red')
        plt.pause(0.1)
    plt.ioff()
    plt.show()
    return np.stack([y, x], axis=1)


from skimage.filters import gaussian
img = cv2.imread('./data/IMG16.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

s = np.linspace(0, 2 * np.pi, 400)
y = 100 + 50 * np.sin(s)  # H
x = 200 + 100 * np.cos(s)  # W
points = np.array([y, x]).T
# plt.figure(figsize=(7, 4), num='snake')
# plt.subplot(1, 2, 1)
# plt.title('a origin circle')
# plt.imshow(img)
# plt.plot(points[:, 1], points[:, 0], linewidth=2, color='red')

res = snake(gaussian(img, 3, preserve_range=False), points, max_iterations=150)
# plt.subplot(1, 2, 2)
# plt.title('after snake')
# plt.imshow(img)
# plt.plot(res[:, 1], res[:, 0], linewidth=2, color='red')
# plt.show()

'''=============level set====================='''

def grad(x, axis=0):
    grad = np.array(np.gradient(x))
    norm_grad = np.sqrt(np.sum(np.power(grad, 2), axis=axis))
    return norm_grad


def F(x, axis=0):
    norm_grad = grad(x, axis)
    return 1./(1. + norm_grad **2)


img = cv2.imread('./data/IMG16.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(7, 4), num='level set')
plt.subplot(1, 2, 1)
plt.title('origin image')
plt.imshow(gray_img, cmap='gray')

# 图像平滑与导数计算
gray_img = gray_img - np.mean(gray_img)
gray_img = cv2.GaussianBlur(gray_img, ksize=(3, 3), sigmaX=0, sigmaY=0)
plt.subplot(1, 2, 2)
plt.title('grad for the image')
plt.imshow(F(gray_img), cmap='gray')
plt.show()


def default_phi(x):
    phi = np.ones(x.shape[:2])
    phi[50:-50, 50:-50] = -1
    return phi

dt = 1
phi = default_phi(gray_img)

plt.ion()
plt.figure(num='level set processing')
plt.clf()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.pause(0.5)

for i in range(150):
    dphi = grad(phi, axis=0)
    dphi_t = F(gray_img) * dphi
    phi = phi + dt*dphi_t
    plt.clf()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.contour(phi, 0)
    plt.pause(0.5)
plt.ioff()
plt.show()


# '''---------插值函数的一些应用小例子----------'''
# import numpy as np
# from scipy.interpolate import RectBivariateSpline
# import matplotlib.pyplot as plt
#
# # Regularly-spaced, coarse grid
# dx, dy = 0.4, 0.4
# xmax, ymax = 2, 2
# x = np.arange(-xmax, xmax, dx)
# print(x)
# y = np.arange(-ymax, ymax, dy)
# X, Y = np.meshgrid(x, y)
#
# Z = -(0.7*X)**2 - (Y/20)**2
# interp_spline = RectBivariateSpline(y, x, Z, kx=2, ky=2, s=0)  # kx/ky确定使用的插值函数的次数，s确定插值函数们是否平滑
#
# # Regularly-spaced, fine grid
# x2 = np.arange(-xmax, xmax, dx).astype(float)
# y2 = np.arange(-ymax, ymax, dy).astype(float)
# X2, Y2 = np.meshgrid(x2, y2)
# Z2 = interp_spline(y2, x2)
#
# print('---对第一个参数求偏导-----')
# print(interp_spline(x2, y2, dx=1, grid=False))
# print('---对第二个参数求偏导-----')
# print(interp_spline(y2, x2, dy=1, grid=False))
#
# fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
# ax[0].plot_wireframe(X, Y, Z, color='k')
#
# ax[1].plot_wireframe(X2, Y2, Z2, color='k')
# for axes in ax:
#     axes.set_zlim(-2, 2)
#     # axes.set_axis_off()
#
# fig.tight_layout()
# plt.show()
