# coding: utf-8
import os
import time
import wandb
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import cv2
import random
from xml.etree.ElementTree import parse
from PIL import Image
from multiprocessing import Process
from my_arguments import WholeSlideArgs
from my_utils import getNetwork, AnalyzeWSI
from my_make_predictions import EvalMetrics
from my_data_augment import WholeSlideTensor


if __name__ == '__main__':
    
    args = WholeSlideArgs().parse()
    net = getNetwork(args)
    
    print("############################\tPreprocessing WSI...")
    SLIDES_PATH = args.slide_path # tiff image directory
    
    for slide_ in sorted(os.listdir(SLIDES_PATH)):
        start = time.time()
        SLIDE = slide_ # full image name
        SLIDE_NAME = slide_.split(".")[0] # get slide name
        
        print(f"\n############################\tSLIDE NAME: {SLIDE}")
        print("############################\t1. Creating Mask...\n")
        
        wsi = AnalyzeWSI(args, SLIDE_NAME)
        patch_list, patch_location, slide_map, stride = wsi.make_mask()
        
        wsi_data = WholeSlideTensor(args,
                                    patch_list=patch_list,
                                    patch_location=patch_location)
        wsi_loader = DataLoader(wsi_data,
                                batch_size=args.wsi_batch_size,
                                shuffle=False,
                                num_workers=args.nWorker)
        
        net.eval()
        tumor_location = []
        
        with torch.no_grad():
            for i, (images, location) in enumerate(wsi_loader):
                img, location = images.cuda(), location.cuda()
                
                out = net(img)
                predicted = torch.argmax(out, dim=1)
                
                for idx, output in enumerate(list(predicted.data)):
                    if output == 1:
                        tumor_location.append(location[idx].cpu().numpy())
            
            tl = str(len(tumor_location))            
            print(f"############################\tNumber of Tumor Patches: {tl}")           
            tumor_location = np.array(tumor_location).astype(int)

            with tqdm(total=len(tumor_location), desc="Color Tumor Patches") as pbar:
                for h,w in tumor_location:
                    cv2.rectangle(slide_map,
                                (stride * h, stride * w),
                                (stride * (h+1), stride * (w+1)),
                                (0, 0, 255),
                                -1)
                
            if not os.path.exists(args.heatmap_saver):
                    os.makedirs(args.heatmap_saver)
                    print(f"############################\tDirectory created: {args.heatmap_saver}")

            cv2.imwrite(f"{args.heatmap_saver}/{SLIDE_NAME}.png", slide_map)
            print(f"############################\t Heatmap Saved at {args.heatmap_saver}")
            print('############################\tTime(s){:.2f}'.format(time.time()-start))
            
            # wsi.plot_heatmap(f"{args.heatmap_saver}/{SLIDE_NAME}.png")
            
