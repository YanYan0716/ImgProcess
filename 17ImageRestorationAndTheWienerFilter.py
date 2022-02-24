'''
reference:https://www.youtube.com/watch?v=GE3_4acUrO4&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=20
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def motion_process(img, motion_angle=60):
    H, W, C = img.shape
    PSF = np.zeros((H, W))
    center_position = (H - 1) / 2
    slope_tan = np.tanh(motion_angle * np.pi / 180)
    slope_cot = 1 / slope_tan

    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF/PSF.sum()


def make_blurred(img, eps=1e-3):
    PSF = motion_process(img)
    b, g, r = cv2.split(img.copy())
    result = []
    for c in [b, g, r]:
        gray_fft = np.fft.fft2(c)
        psf_fft = np.fft.fft2(PSF) + eps
        blurred = np.fft.ifft2(gray_fft*psf_fft)
        blurred = np.abs(np.fft.fftshift(blurred))
        result.append(blurred)
    return result


def wiener(img, eps=1e-3, K=0.01):
    '''
    reference:https://blog.csdn.net/wsp_1138886114/article/details/95024180
    维纳滤波目的：复原图像和原图像的均方误差最小
    :return:
    '''
    PSF = motion_process(img)
    b, g, r = cv2.split(img.copy())
    result = []
    for c in [b, g, r]:
        # 根据傅里叶变换的特性，空间域中的卷积相当于频率域中的乘积

        # 将空间域的图片转换到频率域
        input_fft = np.fft.fft2(c)
        #将噪声转换到频率域
        PSF_fft = np.fft.fft2(PSF)+eps

        PSF_fft = np.conj(PSF_fft)/(np.abs(PSF_fft)**2+K)
        res = np.fft.ifft2(input_fft*PSF_fft)
        # 将结果转回空间域
        res = np.abs(np.fft.fftshift(res))
        result.append(res)
    return result


figure = plt.figure(num='ImageRestorationAndTheWienerFilter', figsize=(6, 3))
img = cv2.imread('./data/IMG21.jpg')
plt.subplot(1, 3, 1)
plt.title('origin')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

blurred_img = make_blurred(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
blurred_img = np.abs(np.transpose(np.array(blurred_img, dtype=np.uint), [1, 2, 0]))
plt.subplot(1, 3, 2)
plt.title('blur')
plt.imshow(blurred_img)


restoration = wiener(np.float32(blurred_img))
restoration = np.abs(np.transpose(np.array(restoration, dtype=np.uint), [1, 2, 0]))
plt.subplot(1, 3, 3)
plt.title('restoration')
plt.imshow(restoration)
plt.show()
