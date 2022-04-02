'''
reference:https://www.youtube.com/watch?v=XTRO6yQOvJc&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=26
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


##===================== graph cut
def GraphCut(img):
    '''
    reference:https://www.youtube.com/watch?v=hJy1kLb9xao
    :param img:
    :return:
    '''
    return img


# ============image inpanting
img = cv2.imread('./data/IMG30.jpg')
figure = plt.figure(num='image blending', figsize=(9, 7))
plt.subplot(2, 2, 1)
plt.title('source image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
