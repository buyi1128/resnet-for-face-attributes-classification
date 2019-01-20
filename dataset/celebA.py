import os
import numpy as np

class CelebA:
    def __init__(self, is_debug):
        self.imgpath = "D:\人脸属性数据集\CelebA\Img\img_align_celeba_png.7z\img_align_celeba_png"
        self.annopath = "D:\人脸属性数据集\CelebA\Anno\list_attr_celeba.txt"
        self.eval_split_path = "D:\人脸属性数据集\CelebA\Eval\list_eval_partition.txt"
        self.is_debug = is_debug
        
    def getAnnoDict(self):
        lines = open(self.annopath).readlines()
        num = lines[0]
        self.attributes = lines[1].strip().split(" ")
        annos = {}
        if self.is_debug:
            lines = lines[2:10]
        else:
            lines = lines[2::]
        for line in lines:
            line = line.strip().split(" ")
            line = [x for x in line if x]
            annos[line[0]] = line[1::]
        return annos
    
    def getImgPath(self, name):
        return os.path.join(self.imgpath, name)

'''
dataset = CelebA(True)
annos = dataset.getAnno()
print(annos)
'''