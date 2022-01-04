'''
reference:
https://www.youtube.com/watch?v=RJEMDkhVgqQ&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=14
https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_morphsnakes.html
https://wenku.baidu.com/view/ce52313181c758f5f71f6767.html
https://blog.csdn.net/cfan927/article/details/108884457
https://blog.csdn.net/cfan927/article/details/108884457
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

'''==========snake(active_contour) ========================'''


from skimage.filters import sobel
def edgedetect(img, type='canny'):
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
    return np.stack([y, x], axis=1)


from skimage.filters import gaussian

img = cv2.imread('1233.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

s = np.linspace(0, 2*np.pi, 400)
y = 150 + 150*np.sin(s)  # H
x = 200 + 180*np.cos(s)  # W
points = np.array([y, x]).T
plt.figure(figsize=(7, 4), num='a origin circle')
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.plot(points[:, 1], points[:, 0], linewidth=2, color='red')

res = snake(gaussian(img, 3, preserve_range=False), points)
plt.subplot(1, 2, 2)
plt.imshow(img)
plt.plot(res[:, 1], res[:, 0], linewidth=2, color='red')
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


# def threepts2circle(pt1, pt2, pt3):
#     '''
#     三点确定一个圆
#     :param pt1:
#     :param pt2:
#     :param pt3:
#     :return: 圆心坐标和半径 h, w, r
#     '''
#     [x1, y1], [x2, y2], [x3, y3] = pt1, pt2, pt3
#     a = x1 - x2
#     b = y1 - y2
#     c = x1 - x3
#     d = y1 - y3
#     e = ((np.power(x1, 2) - np.power(x2, 2)) - (np.power(y2, 2) - np.power(y1, 2))) / 2
#     f = ((np.power(x1, 2) - np.power(x3, 2)) - (np.power(y3, 2) - np.power(y1, 2))) / 2
#     try:
#         h = -(d * e - b * f) / (b * c - a * d)
#         w = -(a * f - c * e) / (b * c - a * d)
#         r = np.sqrt(np.power(x1 - h, 2) + np.power(y1 - w, 2))
#         return h, w, r
#     except:
#         print('something error in function threepts2circle')
#
#
# def circumcircle(points):
#     '''
#     找多个点的最小覆盖圆
#     :param points: list数据 [h, w]
#     :return: 最小覆盖圆的圆心坐标和半径
#     '''
#     center_h, center_w, r = 0, 0, 0
#     past_points = list()
#     for pi in points:
#         if past_points < 2:
#             past_points.append(pi)
#         elif past_points == 2:
#             center_h = int(past_points[0][0] + past_points[1][0]) / 2
#             center_w = int(past_points[1][0] + past_points[1][1]) / 2
#             r = int(np.sqrt(
#                 np.power(past_points[0][0] - past_points[1][0], 2) + np.power(past_points[0][1] - past_points[1][1],
#                                                                               2)))
#         else:
#             # 如果现有最小覆盖圆不包含当前点pt，需要更新最小覆盖圆的坐标等
#             if np.sqrt(np.power(pi[0] - center_h, 2) + np.power(pi[1] - center_w, 2)) > r:
#                 # 找一个新圆
#                 ncenter_h = int(past_points[0][0] + pi[0]) / 2
#                 ncenter_w = int(past_points[0][1] + pi[1]) / 2
#                 nr = int(np.sqrt(np.power(past_points[0][0] - pi[0], 2) + np.power(past_points[0][1] - pi[1], 2)))
#                 pj = None
#                 for pj in past_points:
#                     if np.sqrt(np.power(pj[0] - ncenter_h, 2) + np.power(pj[1] - ncenter_w, 2)) > nr:
#                         ncenter_h = int(pj[0] + pi[0]) / 2
#                         ncenter_w = int(pj[1] + pi[1]) / 2
#                         nr = int(np.sqrt(np.power(pj[0] - pi[0], 2) + np.power(pj[1] - pi[1], 2)))
#                         pass
#                 for pk in past_points:
#                     if np.sqrt(np.power(pk[0] - ncenter_h, 2) + np.power(pk[1] - ncenter_w, 2)) > nr:
#                         center_h, center_w, r = threepts2circle(pi, pj, pk)
#                         pass
#     return center_h, center_w, r