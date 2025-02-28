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

"""
# TODO 1. read github and try staintools
# TODO 2. goal is to weaken the density of purple in original image
# TODO 3. then apply hsv to see how the output changes

# ! need to downsample raw_image -> openslide?
# ! try opencv resize() method -> aple to downsample

# TODO 1) try increase opacity of original tiff image (black marker max opacity)
# * path: '../staintools_output/color_histogram/test1_color_norm_test_1'
"""
Image.MAX_IMAGE_PIXELS = None
METHOD = 'vahadane'
STANDARDIZE_BRIGHTNESS = True
TARGET_PATH = '../target_image/thumbnail_00049.png'

def normalize_stain(raw_path, base_path):
    
    file_list = sorted(os.listdir(raw_path))
    print("Reading Data...")
    
    for file in file_list:
        file_name = file.split('.')[0] + "_color_norm_test_4" # ! update
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
            
        raw_inverted_img = ImageOps.invert(Image.merge("RGB", (r_ch, g_ch, b_ch)))
        thumbnail_coord = raw_inverted_img.getbbox()
        raw_cropped_img = raw_img.crop(thumbnail_coord)
        
        raw_img_path = f"{base_path}/{file_name}/00_original.png"
        raw_cropped_img.save(raw_img_path)
        
        original_img = cv2.imread(raw_img_path)
        height, width = original_img.shape[:2]
        print(f"Image height: {height} | Image width: {width}")
        
        shrink = cv2.resize(original_img, None, fx=0.07, fy=0.07, interpolation=cv2.INTER_AREA)
        s_height, s_width = shrink.shape[:2]
        print(f"Image height: {s_height} | Image width: {s_width}")
        
        shrinked_ver_path = f"{base_path}/{file_name}/01_shrinked.png"
        cv2.imwrite(shrinked_ver_path, shrink)
        

        target_image = staintools.read_image(TARGET_PATH)
        raw_image = staintools.read_image(shrinked_ver_path)
        
        result_dir = f"{base_path}/{file_name}"
        print('-' * 25 + "  Read Target & Raw Image  " + '-' * 25)
        
        images = [target_image, raw_image]
        titles = ["Target"] + ["Original"]
        staintools.plot_image_list(images, width=2, title_list=titles,
                                   save_name=result_dir + "/02_plot_both.png", show=0)
        
        
        if STANDARDIZE_BRIGHTNESS:
            target_image = staintools.LuminosityStandardizer.standardize(target_image)
            raw_image = staintools.LuminosityStandardizer.standardize(raw_image)
            
            images = [target_image, raw_image]
            titles = ["Target Standarized"] + ["Original Standardized"]
            staintools.plot_image_list(images, width=2, title_list=titles,
                                   save_name=result_dir + "/03_standardized.png", show=0)
            
            
        normalizer = staintools.StainNormalizer(method=METHOD)
        normalizer.fit(target_image)
        raw_image_normalized = normalizer.transform(raw_image)
        
        images = [target_image, raw_image_normalized]
        titles = ["Target"] + ["Stain Normalized"]
        staintools.plot_image_list(images, width=2, title_list=titles,
                                   save_name=result_dir + "/04_stain_normalized.png", show=0)
        

        augmentor_target = staintools.StainAugmentor(method=METHOD, sigma1=0.4, sigma2=0.4)
        augmentor_target.fit(target_image)
        augmented_images = []
        
        for _ in range(10):
            augmented_image = augmentor_target.pop()
            augmented_images.append(augmented_image)

        titles = ["Augmented"] * 10
        staintools.plot_image_list(augmented_images, width=5, title_list=titles,
                                   save_name=result_dir + "/05_target_augmented.png", show=0)
        
        
        augmentor_original = staintools.StainAugmentor(method=METHOD, sigma1=0.4, sigma2=0.4)
        augmentor_original.fit(raw_image)
        augmented_images_n = []
        
        for _ in range(10):
            augmented_image = augmentor_original.pop()
            augmented_images_n.append(augmented_image)

        titles = ["Augmented"] * 10
        staintools.plot_image_list(augmented_images_n, width=5, title_list=titles,
                                   save_name=result_dir + "/06_original_augmented.png", show=0)
        
        
        
        augmentor_original = staintools.StainAugmentor(method=METHOD, sigma1=0.4, sigma2=0.4)
        augmentor_original.fit(raw_image_normalized)
        augmented_images_n = []
        
        for _ in range(10):
            augmented_image = augmentor_original.pop()
            augmented_images_n.append(augmented_image)

        titles = ["Augmented"] * 10
        staintools.plot_image_list(augmented_images_n, width=5, title_list=titles,
                                   save_name=result_dir + "/07_original_augmented.png", show=0)
        
        
        print('-' * 25 + "  Function Processed  " + '-' * 25)
        
        
if __name__ == "__main__":
    
    raw_path = '../tesser_image'
    base_path = '../staintools_output/color_histogram'
    
    normalize_stain(raw_path, base_path)