from __future__ import division
import os
import numpy as np
import random
import warnings
try:
    import accimage
except ImportError:
    accimage = None
from PIL import Image, ImageFilter
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

##################################    TRAIN    ##################################
# Mean: [0.76514059 0.54203508 0.7146728]
# Standard Deviation: [0.17992134 0.22158859 0.17864636]

NORM_MEAN = [0.76517, 0.5420, 0.7147]
NORM_STD = [0.1799, 0.2216, 0.1789]

class AugmentData:
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        
        if mode == 'train':
            self.path = args.trainpath
            self.transform = transforms.Compose([
            # image normalization
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
            
            # image direction
            transforms.RandomChoice([transforms.RandomHorizontalFlip(p=1),
                                     transforms.RandomVerticalFlip(p=1)]),
            
            # image color
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))], p=0.1),
            transforms.RandomApply([transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2))], p=0.1),
            transforms.RandomApply([transforms.ColorJitter(saturation=(0.8, 1.2), hue=0.2)], p=0.1)
        ])
            
        elif mode == 'validation':
            self.path = args.validpath
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[:3, :, :]),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
            ])
        else:
            self.path = args.testpath
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[:3, :, :])
                ])
        
        self.lists = self.read_img_and_label()
    
    def read_img_and_label(self):
        """
        Reads text file in data path. Extracts image name and label.

        Returns:
            (list): contains tuple of (image name, label)
        """
        fh = open(self.path, 'r')
        lists = []
        for line in fh:
            line = line.rstrip()
            label_idx = line.rfind(' ')
            if label_idx != -1:
                image_name = line[:label_idx]
                label = int(line[label_idx + 1:])
                lists.append((image_name, label))
        fh.close()
        
        return lists
    
    def __getitem__(self, index):
        """
        Loads augmented data.

        Args:
            index (int): index of dataloader iteration

        Returns:
            img (Image): image from dataset
            label (int): 0 (non-tumor) or 1 (tumor) 
            imagename (str): image number, crop coordinates, label
        """
        image_name, label = self.lists[index]
        img = Image.open(image_name) #.convert('RGB')  
        img = self.transform(img)
        
        return img, label, image_name.split('/')[-1]
    
    def __len__(self):
        """
        Returns:
            (int): length of elements in data path
        """
        return len(self.lists)
    
class WholeSlideTensor:
    def __init__(self, args, patch_list, patch_location):
        self.args = args
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[:3, :, :])
                ])
        
        self.patch_list = patch_list
        self.patch_location = patch_location
        
    def __getitem__(self, index):
        image = self.patch_list[index]
        image = Image.fromarray(np.uint8(image))
        image = self.transform(image)
        
        return image, self.patch_location[index]
    
    def __len__(self):
        """
        Returns:
            (int): length of elements in data path
        """
        return len(self.patch_list)
        
        
