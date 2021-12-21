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
def snake(img):

    return img


img = cv2.imread('./data/IMG14.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = snake(img)