import os
import random
import numpy as np
import PIL.Image as Image
from sklearn.externals import joblib

import param

class CelebA:
    def __init__(self, type):
        self.imgpath = "/home/xyy/data/CelebA/Img/img_align_celeba_png.7z/img_align_celeba_png"
        self.annopath = "/home/xyy/data/CelebA/Anno/list_attr_celeba.txt"
        self.eval_split_path = "/home/xyy/data/CelebA/Eval/list_eval_partition.txt"
        self.type = type
        self.annos = self.getAnnoDict()
        self.mean_pixel = self.getMeanPixel(5)

    def getMeanPixel(self, nPicture):
        pixelFile = os.path.join("dataset/cache", "celebA_meanPixel.txt")
        if os.path.exists(pixelFile):
            meanPixel = np.loadtxt(pixelFile)
            return meanPixel
        keys = list(self.annos.keys())
        n = len(keys)
        sumR = 0
        sumG = 0
        sumB = 0
        for i in range(nPicture):
            selected = random.randint(0, n-1)
            img = Image.open(self.getImgPath(keys[selected]))
            img = np.array(img, dtype=np.float32)
            if len(img.shape) == 2:
                img = np.array([img, img, img])
            sumR = sumR + img[:, :, 0].mean()
            sumG = sumG + img[:, :, 1].mean()
            sumB = sumB + img[:, :, 2].mean()
        meanPixel = np.array([sumR/nPicture, sumG/nPicture, sumB/nPicture])
        np.savetxt(pixelFile, meanPixel)
        return meanPixel

    def getAnnoDict(self):
        if self.type == "train":
            filename = "train_10w.pkl"
        elif self.type == "validation":
            filename = "val_2w.pkl"
        else:
            filename = "test_2w.pkl"
        cacheFile = os.path.join("dataset/cache", filename)
        if os.path.exists(cacheFile):
            return joblib.load(open(cacheFile, "rb"))
        lines = open(self.annopath).readlines()
        self.attributes = lines[1].strip().split(" ")
        annos = {}
        if self.type == "train":
            lines = lines[2:100002]
        elif self.type == "validation":
            lines = lines[100003:120003]
        else:
            lines = lines[140000:160000]
        for line in lines:
            line = line.strip().split(" ")
            line = [x for x in line if x]
            annos[line[0]] = [int(x) for x in line[1::]]
        with open(cacheFile, "wb") as fw:
            joblib.dump(annos, fw)
        return annos
    
    def getImgPath(self, name):
        return os.path.join(self.imgpath, name[:-4]+".png")

        
    

'''
dataset = CelebA(True)
annos = dataset.getAnno()
print(annos)
'''