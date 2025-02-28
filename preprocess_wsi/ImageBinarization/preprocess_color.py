import staintools
import cv2
import time
import numpy as np
import csv
import subprocess
import os
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# ! downsample raw_image -> openslide for efficiency

# TODO 1) analyze staintools for time complexity, space complexity, skip unecessary processes

Image.MAX_IMAGE_PIXELS = None
METHOD = 'vahadane'
STANDARDIZE_BRIGHTNESS = True
TARGET_PATH = '../target_image/thumbnail_best_target.png' # test1_preprocess_5, test3_preprocess_5
# TARGET_PATH = '../target_image/thumbnail_00049.png'

def normalize_stain(raw_path, base_path):
    
    file_list = sorted(os.listdir(raw_path))
    print("Reading Data...")
    
    for file in file_list:
        file_name = file.split('.')[0] + "_preprocess_5" # ! update
        print('-' * 25 + "  Creating File  " + '-' * 25)
        print(f"File Name: {file_name}")
        
        if not os.path.exists(f"{base_path}/{file_name}"):
            os.makedirs(f"{base_path}/{file_name}")
        
        raw_image_file_path = f"{raw_path}/{file}"
        raw_img = Image.open(raw_image_file_path)
        print(f"Image Format: {raw_img.format} | Image Mode: {raw_img.mode} | Image Size: {raw_img.size}")

        if raw_img.mode == 'RGBA':
            r_ch, g_ch, b_ch, alpha_ch = raw_img.split()
        elif raw_img.mode == 'RGB':
            r_ch, g_ch, b_ch = raw_img.split()     
            
        # takes a long time
        raw_inverted_img = ImageOps.invert(Image.merge("RGB", (r_ch, g_ch, b_ch)))
        thumbnail_coord = raw_inverted_img.getbbox()
        raw_cropped_img = raw_img.crop(thumbnail_coord)
        
        raw_img_path = f"{base_path}/{file_name}/00_original.png"
        raw_cropped_img.save(raw_img_path)
        
        original_img = cv2.imread(raw_img_path)
        height, width = original_img.shape[:2]
        print(f"Image height: {height} | Image width: {width}")
        
        
        shrink = cv2.resize(original_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        s_height, s_width = shrink.shape[:2]
        print(f"Image height: {s_height} | Image width: {s_width}")
        
        shrinked_ver_path = f"{base_path}/{file_name}/01_shrinked.png"
        cv2.imwrite(shrinked_ver_path, shrink)

        target_image = staintools.read_image(TARGET_PATH)
        raw_image = staintools.read_image(shrinked_ver_path)
        
        result_dir = f"{base_path}/{file_name}"
        print('-' * 25 + "  Read Target & Raw Image  " + '-' * 25)
        
        
        if STANDARDIZE_BRIGHTNESS:
            target_image = staintools.LuminosityStandardizer.standardize(target_image)
            raw_image = staintools.LuminosityStandardizer.standardize(raw_image)
            
            # images = [target_image, raw_image]
            # titles = ["Target Standarized"] + ["Original Standardized"]
            # staintools.plot_image_list(images, width=2, title_list=titles,
            #                        save_name=result_dir + "/03_standardized.png", show=0)
            
            
        normalizer = staintools.StainNormalizer(method=METHOD)
        normalizer.fit(target_image)
        raw_image_normalized = normalizer.transform(raw_image)
        
        # images = [target_image, raw_image_normalized]
        # titles = ["Target"] + ["Stain Normalized"]
        # staintools.plot_image_list(images, width=2, title_list=titles,
        #                            save_name=result_dir + "/04_stain_normalized.png", show=0)

        print('-' * 25 + "  Start Expanding  " + '-' * 25)
        
        expand = cv2.resize(raw_image_normalized, (width, height), interpolation=cv2.INTER_LANCZOS4)
        e_height, e_width = expand.shape[:2]
        print(f"Image height: {e_height} | Image width: {e_width}")
        
        expanded_ver_path = f"{base_path}/{file_name}/05_expanded.png"
        cv2.imwrite(expanded_ver_path, expand)
        
        print('-' * 25 + "  Function Processed  " + '-' * 25)
        
        
if __name__ == "__main__":
    
    raw_path = '../dcgen_test4' # ! update
    base_path = '../staintools_output/color_modification'
    
    normalize_stain(raw_path, base_path)