import os, sys
import torch.utils.data as D
import cv2
import numpy as np
from torchvision import transforms
import random
import torch
import string, editdistance
sys.path.append("../src/network")
sys.path.append("../src")

class IAM_data(D.Dataset):
    def __init__(self, paths):
        self.img_paths = paths
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __getitem__(self, idx):
        img_path1, img_path2, label, word1, word2 = self.img_paths[idx]
        label = np.float32(label)

        img1 = 255 - cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img1 = self.transforms(img1)

        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE).astype('float32') / 255
        img2 = (img2 - 0.5) / 0.5

        pd_cer, gt_cer = list(word1), list(word2)
        dist = editdistance.eval(pd_cer, gt_cer)
        
        return img1, img2, label, dist
    
    def __len__(self):
        return len(self.img_paths)

