import os
import pickle
import torch
import PIL.Image as Image
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt

from dataset.celebA import CelebA
import param

class MyDataset(Data.Dataset):
    def __init__(self):
        self.batch_size = param.batch_size
        self.is_cuda = param.is_cuda
        self.dataset = CelebA()
        self.annos = self.dataset.annos
        self.nameList = list(self.annos.keys())
        self.imgList = self.getImgs()
        self.meanPixel = self.dataset.mean_pixel
        print("annos", self.annos)
        print("namelist", self.nameList)
        print("meanPixel", self.meanPixel)
        
    def getImgs(self):
        cacheFile = os.path.join("dataset/cache", "debug_images.pkl")
        if os.path.exists(cacheFile):
            return pickle.load(open(cacheFile, "rb"))
        imgList = []
        for name in self.nameList:
            imgfile = self.dataset.getImgPath(name)
            img = Image.open(imgfile)
            imgList.append(img)
        with open(cacheFile, "wb") as fw:
            pickle.dump(imgList, fw)
        return imgList
        
       
    def __len__(self):
        return len(self.nameList)
    
    def __getitem__(self, index):
        img = self.imgList[index]
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32)
        if len(img.shape) == 2:
            img = np.array([img, img, img])
        img[:, :, 0] = img[:, :, 0] - self.meanPixel[0]
        img[:, :, 1] = img[:, :, 1] - self.meanPixel[1]
        img[:, :, 2] = img[:, :, 2] - self.meanPixel[2]
        annos = np.array(self.annos[self.nameList[index]])
        img =torch.Tensor(img)
        img = img.permute(2, 0, 1)
        # img = torch.unsqueeze(img, 0)
        return img, torch.Tensor(annos)