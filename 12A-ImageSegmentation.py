'''
reference:https://www.youtube.com/watch?v=ZF-3aORwEc0&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=13
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import color


'''========basic region growing =================='''
def PixelWithThreshold(img_array, img_pixel, threshold, position):
    '''
    根据某中心像素值，判断其相邻8个像素点是否具有相同颜色
    :param img_array: 图像的数组形式
    :param img_pixel: 中心像素值
    :param threshold: 阈值，大于该阈值则证明两者不相似
    :param position: 中心像素的坐标位置 [h, w]
    :return: 列表，在阈值范围内的像素坐标
    '''
    [H, W] = position

    # 找到center pixel的8个相邻点
    if len(img_array.shape) == 3:
        h, w, c = img_array.shape
    else:
        h, w = img_array.shape
    top, bottom = np.clip(H - 1, 0, h - 1), np.clip(H + 1, 0, h - 1)
    left, right = np.clip(W - 1, 0, w - 1), np.clip(W + 1, 0, w - 1)
    lefttop, ttop, righttop = [top, left], [top, W], [top, right]
    lleft, rright = [H, left], [H, right]
    leftbottom, bbottom, rightbottom = [bottom, left], [bottom, W], [bottom, right]
    neighbors = [lefttop, ttop, righttop, lleft, rright, leftbottom, bbottom, rightbottom]
    neighbors_list = list()

    # 判断8个相邻点中在阈值范围内的即需要的，不在阈值范围内的即要舍弃的
    for neigh in neighbors:
        # 如果是RGB三通道
        if len(img_array.shape) == 3:
            if abs(np.int32(img_array[neigh[0], neigh[1], 0]) - img_pixel[0]) < threshold and \
                    abs(np.int32(img_array[neigh[0], neigh[1], 1]) - img_pixel[1]) < threshold and \
                    abs(np.int32(img_array[neigh[0], neigh[1], 2]) - img_pixel[2]) < threshold:
                neighbors_list.append(neigh)
            else:
                pass
        # 如果是灰度图单通道
        else:
            a = abs(np.int32(img_array[neigh[0], neigh[1]])-img_pixel)
            if a < threshold:
                neighbors_list.append(neigh)
            else:
                pass
    return neighbors_list


def RegionWithThreshold(img_array, img_pixel, threshold, position):
    '''
    根据阈值进行imagesegmentation,
    :param img_array: 图像的数组形式
    :param img_pixel: 中心点像素值
    :param threshold: 阈值范围，范围越大则选取的区域越大
    :param position: 中心点像素的位置
    :return: list(), 在阈值范围内，并且和中心点坐标相连的像素坐标 [h, w]
    '''
    last_list = list()
    now_list = [position]
    while last_list != now_list:
        old_list = list(now_list)
        extra_list = list()
        for point in now_list:
            if point not in last_list:
                neighbor_points = PixelWithThreshold(img_array, img_pixel, threshold, point)
                extra_list += neighbor_points

        new_list = now_list + extra_list
        for i in new_list:
            if not i in now_list:
                now_list.append(i)
        last_list = list(old_list)
    return now_list


def click(event):
    '''
    鼠标点击事件，获取点击的坐标位置和相对应的像素值 进行相应的阈值选择处理
    :param event:
    '''
    (w, h) = event.xdata, event.ydata
    click_position = [int(h), int(w)]
    img_array = np.array(img)
    img_pixel = img_array[click_position[0], click_position[1]]
    points_list = RegionWithThreshold(img_array, img_pixel, threshold=20, position=click_position)
    for i in range(len(points_list)):
        if len(img_array.shape) == 3:
            img_array[points_list[i][0], points_list[i][1]] = [0, 0, 255]
        else:
            img_array[points_list[i][0], points_list[i][1]] = 0
    plt.figure()
    plt.subplot()
    plt.title('image segmentation')
    plt.imshow(img_array, cmap='gray')
    plt.show()


img = cv2.imread('./data/IMG14.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = plt.figure(num='origin image')
cid = fig.canvas.mpl_connect('button_press_event', click)
plt.title('please pick a position from the image, a large range will take more time')
plt.imshow(img, cmap='gray')
plt.show()


'''======k-means and super pixels============='''
def superpixels(img, k, iterator, theta):
    '''
    超像素的生成，最开始的K个cluster是在图片中随机sample的，所以每次的效果可能不一样，仅限于RGB图像，越大的图像计算时间越长
    :param img: 图片数组 [H, W, C]
    :param k: 需要多少个超像素块
    :param iterator: k-means需要迭代duoshaoc
    :param theta: 控制颜色阈和空间阈的比例关系，越大则空间阈对最后的距离影响越大
    :return: 每个超像素块内取rgb均值的图像 [H, W, C]
    '''
    img = cv2.blur(img, (5, 5))
    res_img = np.array(img)

    # 生成k个中心点坐标
    H_img, W_img, _ = img.shape
    s_distance, c_distance = np.power(H_img, 2) + np.power(W_img, 2), 3 * 355 * 255
    h_num = int(np.sqrt(k))
    while k % h_num > 0:
        if H_img > W_img:
            h_num += 1
        else:
            h_num -= 1
    w_num = int(k / h_num)
    h_position = list(np.array(np.random.uniform(0, H_img, h_num), dtype=np.int32))
    w_position = list(np.array(np.random.uniform(0, W_img, w_num), dtype=np.int32))

    position = np.transpose(
        [np.repeat(h_position, len(w_position)), np.tile(w_position, len(h_position))]
    )  # position中的每一个坐标格式为[h, w]

    # 生成k个cluster
    cluster = np.zeros(shape=(len(position), 5))
    for i in range(len(position)):
        cluster[i] = np.array(np.append(img[position[i][0], position[i][1]], position[i]))

    flag_cluster_img = np.zeros(shape=(H_img, W_img))
    for iter in range(iterator):
        # 计算每个像素点和所有cluster的距离
        for h in range(H_img):
            for w in range(W_img):
                pixel_vector = np.array(np.append(img[h, w], [h, w]))
                pixel_Cdistance = np.subtract(pixel_vector[:3], cluster[:, :3])
                pixel_Cdistance = np.sum(np.power(pixel_Cdistance, 2), axis=1) / c_distance
                pixel_Sdistance = np.subtract(pixel_vector[3:], cluster[:, 3:])
                pixel_Sdistance = np.sum(np.power(pixel_Sdistance, 2), axis=1) / s_distance
                pixel_distance = np.sqrt(theta * pixel_Sdistance + pixel_Cdistance)
                near_cluster_ind = np.argmin(pixel_distance)
                flag_cluster_img[h, w] = near_cluster_ind

        for ind in range(k):
            pixs = np.argwhere(flag_cluster_img == ind)
            if pixs.shape[0] == 0:
                pass
            else:
                [h_mean, w_mean] = np.sum(pixs, axis=0) / pixs.shape[0]
                cluster[ind] = np.array(np.append(img[int(h_mean), int(w_mean)], [int(h_mean), int(w_mean)]))

    # 求super pixel的平均值，渲染图片
    for ind in range(k):
        pixs = np.argwhere(flag_cluster_img == ind)
        if pixs.shape[0] == 0:
            pass
        else:
            R, G, B = 0, 0, 0
            for i in range(pixs.shape[0]):
                [r, g, b] = img[pixs[i][0], pixs[i][1]]
                R += r
                G += g
                B += b
            R = int(R / pixs.shape[0])
            G = int(G / pixs.shape[0])
            B = int(B / pixs.shape[0])

            for j in range(pixs.shape[0]):
                res_img[pixs[j][0], pixs[j][1]] = [R, G, B]

    return res_img


img = cv2.imread('./data/IMG15.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

figure = plt.figure(num='k-means and super pixels', figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title('origin image')
plt.imshow(img)

# reference:https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
img2 = img_as_float(img)
segments = slic(img2, n_segments=50, start_label=1, max_iter=10, compactness=30.)
out = color.label2rgb(segments, img2, kind='avg', bg_label=0)
plt.subplot(1, 3, 2)
plt.title('super pixel with skimage')
plt.imshow(out)

super_pixel_img = superpixels(img, k=20, iterator=10, theta=10)
plt.subplot(1, 3, 3)
plt.title('super pixel with myself')
plt.imshow(super_pixel_img)

plt.show()


