import os
import torch
from sklearn.externals import joblib
import PIL.Image as Image
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt

from dataset.celebA import CelebA
import param

class MyDataset(Data.Dataset):
    def __init__(self, type):
        self.batch_size = param.batch_size
        self.is_cuda = param.is_cuda
        self.type = type
        self.dataset = CelebA(self.type)
        self.annos = self.dataset.annos
        self.nameList = list(self.annos.keys())
        self.meanPixel = self.dataset.mean_pixel
        # print("annos", self.annos)
        # print("namelist", self.nameList)
        # print("meanPixel", self.meanPixel)

    def __len__(self):
        return len(self.nameList)
    
    def __getitem__(self, index):
        imgname = self.nameList[index]
        imgfile = self.dataset.getImgPath(imgname)
        img = Image.open(imgfile)
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