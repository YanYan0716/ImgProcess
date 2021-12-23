'''
reference:https://www.youtube.com/watch?v=RJEMDkhVgqQ&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=14
'''
import cv2
import numpy as np

'''
https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_morphsnakes.html
https://wenku.baidu.com/view/ce52313181c758f5f71f6767.html
https://blog.csdn.net/cfan927/article/details/108884457
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from skimage.segmentation import active_contour
from skimage.util import img_as_float
from scipy.interpolate import RectBivariateSpline


'''==========snake(active_contour) ========================'''

def threepts2circle(pt1, pt2, pt3):
    '''
    三点确定一个圆
    :param pt1:
    :param pt2:
    :param pt3:
    :return: 圆心坐标和半径 h, w, r
    '''
    [x1, y1], [x2, y2], [x3, y3] = pt1, pt2, pt3
    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    e = ((np.power(x1, 2)-np.power(x2, 2)) - (np.power(y2, 2)-np.power(y1, 2))) / 2
    f = ((np.power(x1, 2)-np.power(x3, 2)) - (np.power(y3, 2)-np.power(y1, 2))) / 2
    try:
        h = -(d*e - b*f) / (b*c-a*d)
        w = -(a*f - c*e) / (b*c-a*d)
        r = np.sqrt(np.power(x1-h, 2)+np.power(y1-w, 2))
        return h, w, r
    except:
        print('something error in function threepts2circle')


def circumcircle(points):
    '''
    找多个点的最小覆盖圆
    :param points: list数据 [h, w]
    :return: 最小覆盖圆的圆心坐标和半径
    '''
    center_h, center_w, r = 0, 0, 0
    past_points = list()
    for pi in points:
        if past_points < 2:
            past_points.append(pi)
        elif past_points == 2:
            center_h = int(past_points[0][0]+past_points[1][0])/2
            center_w = int(past_points[1][0]+past_points[1][1])/2
            r = int(np.sqrt(np.power(past_points[0][0]-past_points[1][0], 2)+np.power(past_points[0][1]-past_points[1][1], 2)))
        else:
            # 如果现有最小覆盖圆不包含当前点pt，需要更新最小覆盖圆的坐标等
            if np.sqrt(np.power(pi[0]-center_h, 2)+np.power(pi[1]-center_w, 2)) > r:
                # 找一个新圆
                ncenter_h = int(past_points[0][0]+pi[0])/2
                ncenter_w = int(past_points[0][1]+pi[1])/2
                nr = int(np.sqrt(np.power(past_points[0][0]-pi[0], 2)+np.power(past_points[0][1]-pi[1], 2)))
                pj = None
                for pj in past_points:
                    if np.sqrt(np.power(pj[0]-ncenter_h, 2)+np.power(pj[1]-ncenter_w, 2)) > nr:
                        ncenter_h = int(pj[0] + pi[0]) / 2
                        ncenter_w = int(pj[1] + pi[1]) / 2
                        nr = int(np.sqrt(np.power(pj[0] - pi[0], 2) + np.power(pj[1] - pi[1], 2)))
                        pass
                for pk in past_points:
                    if np.sqrt(np.power(pk[0] - ncenter_h, 2) + np.power(pk[1] - ncenter_w, 2)) > nr:
                        center_h, center_w, r = threepts2circle(pi, pj, pk)
                        pass
    return center_h, center_w, r


def edgedetect(img, type='canny'):
    edge_img = np.zeros(shape=img.shape)
    if type == 'canny':
        img = cv2.GaussianBlur(img, (5, 5), 0)
        edge_img = cv2.Canny(img, 50, 150)
    return edge_img


def snake(img, snake, w_edge=0, w_line=1, coordinates='rc'):
    snake_xy = snake
    x, y = snake_xy[:, 0].astype(np.float), snake_xy[:, 1].astype(np.float)
    convergence_order = 10
    n = len(x)
    xsave = np.empty((convergence_order, n))
    ysave = np.empty((convergence_order, n))


    edge_img = edgedetect(img)
    res = RectBivariateSpline(
        np.arange(edge_img.shape[1]),
        np.arange(edge_img.shape[0]),
        edge_img.T,
        kx=2,
        ky=2,
        s=0
    )

    X, Y = np.meshgrid(np.arange(0, edge_img.shape[0], 1), np.arange(0, edge_img.shape[1], 1))
    fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
    ax[0].plot_wireframe(X, Y, edge_img[X, Y], color='r')
    # for axes in ax:
    #     axes.set_zlim(-0.2, 1)
        # axes.set_axis_off()

    fig.tight_layout()
    plt.show()

    return img


img = cv2.imread('./data/IMG14.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

s = np.linspace(0, 2*np.pi, 100)
init = 50 * np.array([np.sin(s), np.cos(s)]).T
plt.figure(figsize=(5, 5), num='a origin circle')
plt.plot(init.T[0], init.T[1], linewidth=2, color='red')
plt.show()


snake(img, init, w_edge=0, w_line=1, coordinates='rc')
#
# import numpy as np
# from scipy.interpolate import RectBivariateSpline
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# #
# # # Regularly-spaced, coarse grid
# dx, dy = 0.4, 0.4
# xmax, ymax = 2, 4
# x = np.arange(-xmax, xmax, dx)
# y = np.arange(-ymax, ymax, dy)
# X, Y = np.meshgrid(x, y)
# Z = np.exp(-(2*X)**2 - (Y/2)**2)
#
# interp_spline = RectBivariateSpline(y, x, Z)
#
# # # Regularly-spaced, fine grid
# dx2, dy2 = 0.1, 0.1
# x2 = np.arange(-xmax, xmax, dx2)
# y2 = np.arange(-ymax, ymax, dy2)
# X2, Y2 = np.meshgrid(x2, y2)
# Z2 = interp_spline(y2, x2)
# np.set_printoptions(suppress=True)
#
# fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
# ax[0].plot_wireframe(X, Y, Z, color='k')
#
# ax[1].plot_wireframe(X2, Y2, Z2, color='k')
# for axes in ax:
#     axes.set_zlim(-0.2,1)
#     # axes.set_axis_off()
#
# fig.tight_layout()
# plt.show()

