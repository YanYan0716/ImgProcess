
'''
reference:https://www.youtube.com/watch?v=EiD0PuL2yhk&list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX&index=18
'''
import cv2
import numpy as np

'''=====哈夫曼编码======='''


class Node(object):
    def __init__(self, name=None, value=None):
        self._name = name
        self._value = value
        self._left = None
        self._right = None


class HuffmanTree(object):
    '''
    reference:https://zhuanlan.zhihu.com/p/108845114
    '''
    def __init__(self, char_weights):
        '''
        定义哈夫曼树类
        :param char_weights:
        '''
        self.Leav = [Node(part[0], part[1]) for part in char_weights]
        while len(self.Leav) != 1:
            self.Leav.sort(key=lambda node: node._value, reverse=True)
            c = Node(value=(self.Leav[-1]._value + self.Leav[-2]._value))
            c._left = self.Leav.pop(-1)
            c._right = self.Leav.pop(-1)
            self.Leav.append(c)
        self.root = self.Leav[0]
        self.Buffer = list(range(100))

    def pre(self, tree, length, encoding):
        '''
        递归进行哈夫曼编码
        :param tree:
        :param length:
        :param encoding:
        :return:
        '''
        node = tree
        if not node:
            return
        elif node._name:
            encoding.append((node._name, self.Buffer[:length]))
        self.Buffer[length] = 0
        self.pre(node._left, length + 1, encoding)
        self.Buffer[length] = 1
        self.pre(node._right, length + 1, encoding)

    def get_code(self):
        encoding = list()
        self.pre(self.root, 0, encoding)
        return encoding


def Huffman(img):
    '''
    灰度图的哈夫曼编码
    :param img:
    :return: 各个像素值的编码结果 list形式
    '''
    [H, W] = img.shape
    probability = np.zeros(shape=(256))

    for h in range(H):
        for w in range(W):
            probability[img[h, w]] += 1

    info = list()
    for i in range(len(probability)):
        info.append((str(i), probability[i] / (H * W)))

    # 获取哈夫曼编码
    huffmantree = HuffmanTree(info)
    encoding = huffmantree.get_code()
    return encoding


img = cv2.imread('./data/IMG18.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
encode_str = Huffman(gray_img)
for (pixel, code) in enumerate(encode_str):
    print(code[0] + '  code: ' + str(code[1]))


