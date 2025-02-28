import os
import sys
import numpy as np
import random
import torch
import cv2
import openslide
import random
from xml.etree.ElementTree import parse
from PIL import Image
from multiprocessing import Process
import matplotlib.pyplot as plt

import my_networks


############################## THIS IS FOR TRAIN, TEST, WSI ANALYSIS
def getNetwork(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    net = getattr(my_networks, args.network)(in_channels=3, num_classes=args.nCls)
    
    if len(args.gpu_ids) > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()
    
    # * LOAD FINETUNE WEIGHT PARAMETERS (DEFAULT: FALSE)
    if args.is_pretrained:
        try:
            if os.path.exists(args.test_restore):
                print('####################\tLoading model...', args.test_restore)
                net.load_state_dict(torch.load(args.test_restore))
                print('####################\tModel Successfully Loaded...')
        except:
            print('####################\tFailed to load pre-trained network...')
            sys.exit()

    return net

############################## THIS IS FOR WSI ANALYSIS
class AnalyzeWSI:
    def __init__(self, args, slide_name):
        self.args = args
        self.slide_name = slide_name
    
    def get_dir(self, flags):
        if flags == "slide":
            return f"{self.args.slide_path}/{self.slide_name}.tif"

    def make_mask(self):
        slide_path = self.get_dir("slide")
        mask_level = self.args.mask_level
        tissue_threshold = self.args.tissue_thres
        patch_size = int(self.args.patch_size)
        
        try:
            slide = openslide.OpenSlide(slide_path)
            # slide = openslide.OpenSlide(f"{slide_path}/{slide_name}.tif")
        except openslide.OpenSlideError:
            print(f"Error: Cannot Open OpenSlide Module. {slide_path}")
            sys.exit()
            
        slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[mask_level]))
        
        slide_lv = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
        slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
        slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
        slide_lv = slide_lv[:, :, 1]
        _, tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tissue_mask = np.array(tissue_mask)
        
        patch_size = patch_size
        width, height = np.array(slide.level_dimensions[0]) // patch_size
        total = width * height
        num_all_patches, num_tissue_patch = 0, 0
        stride = int(patch_size / (2**mask_level))
        print(f"total : {total} | patch_size : {patch_size} | stride : {stride}")
        
        patch_list = [] # this is actual regions for wsi (in np format)
        patch_location = [] # x-y coordinates of patches
        
        for i in range(width):
            for j in range(height):
                tissue_mask_sum = tissue_mask[stride * j : stride * (j + 1),
                                            stride * i : stride * (i + 1)].sum()
                tissue_mask_max = stride * stride * 255
                tissue_area_ratio = tissue_mask_sum / tissue_mask_max

                if tissue_area_ratio > tissue_threshold:
                    patch = np.array(slide.read_region((patch_size * i, patch_size * j),
                                                    0,
                                                    (patch_size, patch_size)))
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
                    patch_list.append(patch)
                    patch_location += [[i, j]]
                    num_tissue_patch += 1
                
                num_all_patches +=1
                
        print('All Patch: %d, Tissue Patch: %d' % (
                    num_all_patches,
                    num_tissue_patch))

        patch_list = np.array(patch_list)
        patch_location = np.array(patch_location)
        
        return patch_list, patch_location, slide_map, stride


    def plot_heatmap(self, heatmap_path):
        
        # patch_map = cv2.imread(f"{cf.mask_path}/b_{SLIDE}/b_{SLIDE}_map_patch.png")
        prediction_map = cv2.imread(f"{heatmap_path}")

        plt.figure(figsize=(40, 40))
        # plt.subplot(141), plt.axis('off'), plt.imshow(patch_map)
        # plt.title("Patch Map")
        plt.subplot(142), plt.axis('off'), plt.imshow(prediction_map)
        plt.title("Prediction Result")
        
        return None