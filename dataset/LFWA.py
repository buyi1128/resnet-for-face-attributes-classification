import os
import numpy as np


class LFWA:
    def __init__(self, is_debug):
        self.imgpath = "D:\人脸属性数据集\lfw-deepfunneled"
        self.annopath = "D:\人脸属性数据集\lfw_attributes.txt"
        self.is_debug = is_debug

'''
dataset = LFWA(True)
annos = dataset.getAnno()
print(annos)
'''