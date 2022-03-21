'''
reference:https://www.youtube.com/watch?v=O2RwWHWHQlM&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX
'''

'''
reference:https://www.youtube.com/watch?v=UJtV3DdjCVY&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=23&t=2s
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


def SpatialWatermark(img1, img2, alpha):
    new_img = (1-alpha)*img1+alpha*img2
    new_img = new_img.astype(img1.dtype)
    return new_img


def LeastSignifitionBits(img1, img2, bits=4):
    pass

img1 = cv2.imread('./data/IMG24.jpg')
figure = plt.figure(num='dither', figsize=(9, 6))
plt.subplot(2, 3, 1)
plt.title('image1')
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

img2 = cv2.imread('./data/IMG25.jpg')
plt.subplot(2, 3, 2)
plt.title('image2')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

result = SpatialWatermark(img1, img2, alpha=0.3)
plt.subplot(2, 3, 3)
plt.title('SpatialWatermark')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

result = SpatialWatermark(img1, img2, alpha=0.3)
plt.subplot(2, 3, 4)
plt.title('make the least significant bits')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))



plt.show()
