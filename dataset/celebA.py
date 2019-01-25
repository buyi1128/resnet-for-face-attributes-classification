import os
import random
import numpy as np
import PIL.Image as Image
import pickle

import param

class CelebA:
    def __init__(self):
        self.imgpath = "D:\人脸属性数据集\CelebA\Img\img_align_celeba_png.7z\img_align_celeba_png"
        self.annopath = "D:\人脸属性数据集\CelebA\Anno\list_attr_celeba.txt"
        self.eval_split_path = "D:\人脸属性数据集\CelebA\Eval\list_eval_partition.txt"
        self.is_debug = param.is_debug
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
        cacheFile = os.path.join("dataset/cache", "debug_annos.pkl")
        if os.path.exists(cacheFile):
            return pickle.load(open(cacheFile, "rb"))
        lines = open(self.annopath).readlines()
        self.attributes = lines[1].strip().split(" ")
        annos = {}
        if self.is_debug:
            lines = lines[2:10]
        else:
            lines = lines[2::]
        for line in lines:
            line = line.strip().split(" ")
            line = [x for x in line if x]
            annos[line[0]] = [int(x) for x in line[1::]]
        with open(cacheFile, "wb") as fw:
            pickle.dump(annos, fw)
        return annos
    
    def getImgPath(self, name):
        return os.path.join(self.imgpath, name[:-4]+".png")

        
    

'''
dataset = CelebA(True)
annos = dataset.getAnno()
print(annos)
'''