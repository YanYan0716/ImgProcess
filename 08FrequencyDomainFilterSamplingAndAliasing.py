'''
reference:https://www.youtube.com/watch?v=tmERrPh1E4c&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=8&t=18s
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

# fig, ax = plt.subplots(3, 2, figsize=(8, 10), num='fft with different image')
# img = cv2.imread('./data/IMG1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.subplot(321)
# plt.title('original img')
# plt.imshow(img, cmap='gray')
#
# dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
# plt.subplot(322)
# plt.title('magnitude_spectrum')
# plt.imshow(magnitude_spectrum, cmap='gray')
#
# out = np.abs(dft)
# out[1, 1, 0] = 0
# dft_shift = np.fft.fftshift(out)
# out = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
#
# plt.subplot(323)
# plt.imshow(out, cmap='gray')
# print('the max of out: ' + str(np.max(out)))
#
# ax1 = plt.subplot(324)
# plt.title('hist about out')
# ax1.hist(out.ravel(), bins=20, color='black')
#
# plt.subplot(325)
# plt.title('clip out [0, 500000]')
# plt.imshow(np.clip(out, 0, 500000), cmap='gray')
#
# plt.show()


def gauss_filter(kernel_size, sigma=1):
    kx = cv2.getGaussianKernel(kernel_size[0], sigma)
    ky = cv2.getGaussianKernel(kernel_size[1], sigma)
    return np.multiply(kx, ky.T)


def idealfilter(im, edges, type='lp'):
    '''
    图像滤波
    :param im: 图片矩阵
    :param edges: 滤波边长，一个数表示圆形直径，两个数表示方形长宽
    :param type: 滤波种类，有低频滤波 高斯滤波 拉普拉斯滤波等
    :return: 无
    '''
    im = im.astype(np.float)
    mask = np.zeros(shape=im.shape)
    M, N = im.shape
    rM = int(np.round(M / 2) + 1)
    rN = int(np.round(N / 2) + 1)
    if len(edges) < 2:
        x, y = np.meshgrid(np.arange(1, im.shape[0] + 1), np.arange(1, im.shape[1] + 1))
        ind = np.transpose((np.square(x - rM) + np.square(y - rN)) < np.square(edges))
        mask = mask + ind
    else:
        mask[rM - edges[0]:rM + edges[0], rN - edges[1]:rN + edges[1]] = 1
    mask = np.fft.fftshift(mask)

    # if type == 'hp':
    #     mask = mask.astype(np.bool)
    #     mask = ~mask

    if type == 'gausslp':
        h = gauss_filter(kernel_size=[M, N], sigma=edges[0])
        mask = np.abs(np.fft.ifft2(h))
    #
    # if type == 'laplacian':
    #     h = -4 * np.power(np.pi, 2) * np.transpose((np.power((x - rM), 2) + np.power((y - rN), 2)))
    #     mask = np.fft.fftshift(h)

    # 显示结果
    fig, ax = plt.subplots(3, 2, figsize=(8, 10), num='filter with images')
    plt.subplot(321)
    plt.title('original image')
    plt.imshow(im, cmap='gray')

    dft = np.fft.fft2(im)
    dft_shift = np.fft.fftshift(dft)
    fft = np.log(np.abs(dft_shift))
    plt.subplot(322)
    plt.title('original FFT')
    plt.imshow(fft, cmap='gray')

    plt.subplot(323)
    plt.title('mask')
    plt.imshow(np.fft.fftshift(mask), cmap='gray')

    plt.subplot(324)
    plt.title('output FFT')
    ftout = mask * dft
    plt.imshow(np.log(np.clip(np.abs(np.fft.fftshift(ftout)), a_min=1, a_max=np.inf)), cmap='gray')

    plt.subplot(325)
    plt.title('filtered image with ringing')
    ftout_ = np.fft.ifftshift(ftout)
    out = np.abs(np.fft.ifft2(ftout_))
    plt.imshow(out, cmap='gray')

    # plt.subplot(326)
    # plt.title('Spatial filter co to mask')
    # out = np.abs(np.fft.ifft2(mask))
    # plt.imshow(out, cmap='gray')
    plt.show()

    x, y = np.arange(int(N)), np.arange(int(M))
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, np.abs(np.fft.fftshift(ftout)), cmap=plt.get_cmap('rainbow'), rstride=2, cstride=2)
    plt.title('Spatial filter as surface')
    plt.show()

    return 0


im = cv2.cvtColor(cv2.imread('./data/IMG1.jpg'), cv2.COLOR_BGR2GRAY)
idealfilter(im, [20, 20], 'lp')
idealfilter(im, [20], 'lp')


def filterft(h):

    ft = np.fft.fft2(h, s=[512, 512])
    fft = np.log(np.abs(ft))
    ft_shift = np.fft.fftshift(fft)
    plt.title('original FFT')
    plt.imshow(ft_shift, cmap='gray')
    plt.show()

    x, y = np.arange(int(512)), np.arange(int(512))
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, np.fft.fftshift(np.abs(ft)), cmap=plt.get_cmap('rainbow'), rstride=2, cstride=2)
    plt.title('fourier domain of h')
    plt.show()


f = np.ones(shape=(3, 3))/9
filterft(f)
f = np.ones(shape=(30, 30))/900
filterft(f)

idealfilter(im, [20], 'gausslp')