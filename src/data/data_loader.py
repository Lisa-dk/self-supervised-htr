import os, sys
import torch.utils.data as D
import cv2
import numpy as np



class RIMES_data(D.Dataset):
    def __init__(self, paths, input_size, tokenizer, num_images):
        self.img_paths = paths
        self.input_size = input_size
        self.tokenizer = tokenizer
        self.num_images = num_images


    def img_padding(self, img, input_height):
        # split by inverted because otherwise images with small height get 
        # too 'zoomed in'
        if len(img)/input_height <= 0.5:
            desired_height = input_height
            delta_h = int(((desired_height - len(img))/2) * (1 - (len(img)/input_height)))
        else:
            delta_h = 0
        new_im = np.pad(
                    img,
                    pad_width=((delta_h, delta_h), (0, 0)),
                    mode="constant",
                    constant_values=(255),
        )
        return new_im
    

    def read_image_single(self, file_name):
        url = os.path.join(file_name)
        img = cv2.imread(url, 0)

        if img is None and os.path.exists(url):
            # image is present but corrupted
            return np.zeros((self.input_size[0], self.input_size[1])), 0
        
        img = self.img_padding(img, self.input_size[0])

        u, i = np.unique(np.array(img).flatten(), return_inverse=True)
        background = int(u[np.argmax(np.bincount(i))])

        wt, ht, = self.input_size[1], self.input_size[0]
        h, w = np.asarray(img).shape

        # resize maintaining ratio
        f = max((w / wt), (h / ht))
        new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))

        img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
        
        img_width = img.shape[-1]

        outImg = np.ones((self.input_size[0], self.input_size[1]), dtype='float32') * background
        start = int((ht / 2) - (new_size[1] / 2))
        end = start + new_size[1]
        outImg[start:end, :img_width] = img
        outImg = outImg.astype('float32')

        return outImg, img_width
    
    def normalize(self, img):
        img = 1. - (img / 255.) # 0-255 -> 0-1

        m, s = 0.5, 0.5
        return (img - m) / s

    
    def copy_style_imgs(self, single_imgs):
        idx = 0
        final_imgs = []

        for idx in range(len(single_imgs)):
            imgs = list()
            img = single_imgs[idx]
            imgs.append(img)

            final_img = imgs

            while len(final_img) < self.num_images:
                num_cp = self.num_images - len(final_img)
                final_img = final_img + imgs[:num_cp]

            final_img = np.stack(final_img, axis=0)
            final_imgs.append(final_img)
            idx += 1

        return np.asarray(final_imgs)
    
    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        print(img.shape, img_path)

        img = self.normalize(img)
        # gen_input = self.copy_style_imgs(img)

        label = self.tokenizer.encode(label)
        
        return img, label
    
    def __len__(self):
        return len(self.img_paths)

