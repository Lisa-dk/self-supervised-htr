import os, sys
import torch.utils.data as D
import cv2
import numpy as np
from torchvision import transforms
import random
import torch

class RIMES_data(D.Dataset):
    def __init__(self, paths, input_size, tokenizer, num_images):
        self.img_paths = paths
        self.input_size = input_size
        self.tokenizer = tokenizer
        self.num_images = num_images

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def copy_style_imgs(self, single_img, img_path):
        """Obtain different same-style images from a directory"""
        img_dir ="\\".join(img_path.split('\\')[:-1])
        style_paths = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]
        copy_paths = random.choices(style_paths, k=self.num_images - 1)

        final_img = [single_img[:]]

        for path in copy_paths:
            new_img = 255 - cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
            new_img = self.transforms(new_img)
            final_img = final_img + [new_img]

        final_img = np.stack(final_img, axis=0)

        return np.asarray(final_img, dtype="float32")

    
    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]

        img = 255 - cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        img = self.transforms(img)
        gen_input = self.copy_style_imgs(img, img_path)

        label = self.tokenizer.encode(label)
        
        return img, gen_input, np.asarray(label)
    
    def __len__(self):
        return len(self.img_paths)

