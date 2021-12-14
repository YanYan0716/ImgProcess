'''
reference:https://www.youtube.com/watch?v=ojapO75FV38&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=12&t=2009s
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def SelfOtus(img):
    h, w = img.shape
    global_P = np.zeros(shape=[256])
    for i in range(h):
        for j in range(w):
            global_P[int(img[i, j])] += 1
    global_P = global_P/(h*w)
    intensity = np.arange(0, 256, 1)
    global_mean = np.sum(np.multiply(intensity, global_P))
    global_var = np.sum(np.multiply(np.power(intensity-global_mean, 2), global_P))

    flag_ratio = 0
    flag_intensity = 0
    for i in range(len(intensity-1)):
        p1 = np.sum(global_P[:i+1])
        p2 = 1 - p1
        m1 = np.sum(np.multiply(intensity[:i+1], global_P[:i+1])) / p1
        m2 = np.sum(np.multiply(intensity[i+1:], global_P[i+1:])) / p2
        part_var = np.sum(p1*np.power((m1-global_mean), 2))+np.sum(p2*np.power((m2-global_mean), 2))
        ratio = part_var/global_var
        if flag_ratio < ratio:
            flag_ratio = ratio
            flag_intensity = i

    new_img = np.where(img > flag_intensity, 255, 0)
    return new_img, flag_intensity

fig, ax = plt.subplots(2, 2, figsize=(10, 10), num='canny')
img = cv2.imread('./data/IMG12.jpg')
plt.subplot(221)
plt.title('original img')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(222)
plt.title('gray img')
plt.imshow(gray_img, cmap='gray')


intensity, cv2otsu_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
plt.subplot(223)
plt.title('otsu in opencv, intensity: '+str(intensity))
plt.imshow(cv2otsu_img, cmap='gray')


selfotsu_img, intensity = SelfOtus(gray_img)
plt.subplot(224)
plt.title('otsu by myself, intensity: '+str(intensity))
plt.imshow(selfotsu_img, cmap='gray')
plt.show()