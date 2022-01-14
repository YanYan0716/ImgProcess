'''
reference:https://www.youtube.com/watch?v=ddXvs1Wp95A&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=17&t=7s
'''


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def templateMatch(template, image, threshold):
    '''
    模板匹配算法
    :param template:
    :param image:
    :param threshold:
    :return:
    '''
    [Ht, Wt] = template.shape
    [Hi, Wi] = image.shape

    temp_mean = np.mean(template)
    temp_ = (template - temp_mean)/np.sqrt(np.sum(np.power(template-temp_mean, 2)))
    score = list()

    for h in range(0, Hi-Ht, 1):
        for w in range(0, Wi-Wt, 1):
            window = image[h: h+Ht, w:w+Wt]
            win_mean = np.mean(window)
            win_ = (window - win_mean)/np.sqrt(np.sum(np.power(window-win_mean, 2)))
            score_ = np.sum(temp_*win_)
            if score_ > threshold:
                score.append([score_, h, w])
    return score


figure = plt.figure(num='template match', figsize=(9, 3))
img = cv2.imread('./img.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(1, 3, 1)
plt.title('origin')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

temp = cv2.imread('./temp.JPG')
gray_temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
plt.subplot(1, 3, 2)
plt.title('template')
plt.imshow(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))

res = templateMatch(gray_temp, gray_img, 0.9)
[Ht, Wt] = gray_temp.shape
plt.subplot(1, 3, 3)
plt.title('match')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

for i in range(len(res)):
    rect = patches.Rectangle((res[i][1], res[i][2]), Ht, Wt, linewidth=1, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)

plt.show()
