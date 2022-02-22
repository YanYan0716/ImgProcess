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
    b_gray, g_gray, r_gray = cv2.split(img.copy())
    result = []
    for gray in [b_gray, g_gray, r_gray]:
        gray_fft = np.fft.fft2(gray)
        psf_fft = np.fft.fft2(PSF) + eps
        blurred = np.fft.ifft2(gray_fft*psf_fft)
        blurred = np.abs(np.fft.fftshift(blurred))
        result.append(blurred)
    return result


def wiener():
    pass


figure = plt.figure(num='ImageRestorationAndTheWienerFilter', figsize=(6, 6))
img = cv2.imread('./data/IMG21.jpg')
plt.subplot(2, 2, 1)
plt.title('origin')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

PSF = make_blurred(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
PSF = np.abs(np.transpose(np.array(PSF, dtype=np.uint), [1, 2, 0]))
print(PSF)
plt.subplot(2, 2, 2)
plt.title('blur')
plt.imshow(PSF)
plt.show()
