import time
import numpy as np
# import csv
import cv2
from tqdm import tqdm
import skimage.morphology
import subprocess
import os
import tifffile as tiff
# import openslide
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
DCGEN_PATH = '../dc_cropped_ver'
# Image.MAX_MEMORY_USAGE = 0 

def filter_small_components(img, min_size):
    labeledc, numc = skimage.morphology.label(img, return_num=True)
    component_sizes = np.unique(labeledc, return_counts=True)
    sel_components = component_sizes[0][component_sizes[1] >= min_size]
    img_filtered = np.zeros(img.shape[:2], dtype=np.uint8)
    img = img.astype(np.uint8)
    
    for i in sel_components:
        comp_indices = (labeledc==i)
        img_filtered[comp_indices] = img[comp_indices]
        
    return img_filtered


def get_large_components(img):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    connectivity = 8
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    areas = stats[:, -1]

    max_label = 1
    
    if len(areas) > 1:
        max_area = areas[1]
    else:
        max_area = areas[0]
        
    for i in range(2, num_labels):
        if areas[i] > max_area:
            max_label = i
            max_area = areas[i]

    large_component = np.zeros_like(labels)
    large_component[labels == max_label] = 255
    
    return large_component


def image_binarization(raw_path, base_path):
    print('-' * 25 + "  Start penAnnotation  " + '-' * 25)
    print('-' * 25 + " Locating File Directory  " + '-' * 25)
    
    file_list = sorted(os.listdir(raw_path))

    for file in tqdm(file_list, desc="Annotation penMarker"):
        
        file_name = file.split('.')[0] # preprocessed_test1
        test_num = file_name.split('_')[1] # test1
        tiff_name = "cropped_" + test_num + ".tif" # cropped_test1.tif
        print('-' * 25 + "Creating File" + '-' * 25)
        print(f"File Name: {file_name}")
        print(f"Test Name (file): {test_num}")
        print(f"Tiff Name: {tiff_name}")
        
        if not os.path.exists(f"{base_path}/{test_num}/step"):
            os.makedirs(f"{base_path}/{test_num}/step")
    
        # * hyperparameter
        dilation_radius = 89
        min_component_size = 50000
        
        raw_img_path = f"{raw_path}/{file}"        
        penMarker_img = cv2.imread(raw_img_path)
        penMarker_img_blur = cv2.GaussianBlur(penMarker_img, (55,55), 0)
        
        penMarker_img_blur_path = f"{base_path}/{test_num}/step/01_blur.png"
        cv2.imwrite(penMarker_img_blur_path,  penMarker_img_blur)
        
        img_hsv = cv2.cvtColor(penMarker_img_blur, cv2.COLOR_BGR2HSV)
        penMarker_img_hsv_path = f"{base_path}/{test_num}/step/02_hsv.png"
        cv2.imwrite(penMarker_img_hsv_path, img_hsv)
        
        tissue_hsv = cv2.inRange(img_hsv, np.array([135, 10, 30]), np.array([170, 255, 255]))
        tissue_img_path = f"{base_path}/{test_num}/step/03_cell_tissue.png"
        cv2.imwrite(tissue_img_path, tissue_hsv)
        
        # save cell info
        mask_tissue = tissue_hsv

        hsv_black_marker = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255, 125]))
        black_marker_path = f"{base_path}/{test_num}/step/04_hsv_black_marker.png"
        cv2.imwrite(black_marker_path, hsv_black_marker)

        dilated_mask = cv2.dilate(hsv_black_marker, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius, dilation_radius)))
        dilated_mask_path = f"{base_path}/{test_num}/step/05_marker_dilated.png"
        cv2.imwrite(dilated_mask_path, dilated_mask)
        
        
        marker_contours = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marker_enclosed_mask = np.zeros(dilated_mask.shape)
        print(f"Filling Contour....")
        
        for cidx in tqdm(range(len(marker_contours[0])), desc='Marker Contours'):
            cv2.drawContours(marker_enclosed_mask, marker_contours[0], cidx, (255,255,255), thickness=-1)
        print('-' * 25 + " Contour Filled  " + '-' * 25)
           
            
        marker_enclosed_mask_path = f"{base_path}/{test_num}/step/06_contour_filled.png"
        cv2.imwrite(marker_enclosed_mask_path,  marker_enclosed_mask)
        print('-' * 25 + " Contour Successfully Saved  " + '-' * 25)
        
        # filter noise
        mask_filtered = filter_small_components(marker_enclosed_mask, min_component_size)
        print('-' * 25 + " Filtered Small Components  " + '-' * 25)
        mask_filtered_path = f"{base_path}/{test_num}/step/07_contour_noise_filtered.png"
        cv2.imwrite(mask_filtered_path,  mask_filtered)
        
        # get tumorous cell chunk
        large_mask = get_large_components(mask_filtered)
        print('-' * 25 + " Get Large Mask  " + '-' * 25)
        large_mask_path = f"{base_path}/{test_num}/step/08_mask.png"
        cv2.imwrite(large_mask_path, large_mask)
 
 
        # invert large mask (get non-tumor region)
        large_mask_ = Image.fromarray(large_mask)
        large_mask_ = large_mask_.convert("L")
        inverted_large_mask = ImageOps.invert(large_mask_)
        inverted_large_mask_path = f"{base_path}/{test_num}/step/09_inverted_mask.png"
        cv2.imwrite(inverted_large_mask_path, inverted_large_mask)

        # large_marker = get_marker_line(hsv_black_marker)
        large_marker = get_large_components(dilated_mask)
        large_marker_path = f"{base_path}/{test_num}/step/10_large_marker.png"
        cv2.imwrite(large_marker_path, large_marker)
        
        subtract_img = large_mask - large_marker
        subtract_img = np.uint8(subtract_img)
        print('-' * 25 + " Get tumor Region  " + '-' * 25)
        subtract_img_path = f"{base_path}/{test_num}/step/11_subtract_img.png"
        cv2.imwrite(subtract_img_path, subtract_img)

        # filter noise -> may be possible to use filter_small_components
        filtered_subtract_img = get_large_components(subtract_img)
        filtered_sub_path = f"{base_path}/{test_num}/step/12_mask_subtract_marker.png"
        cv2.imwrite(filtered_sub_path, filtered_subtract_img)
        
        filtered_subtract_img = filtered_subtract_img.astype(np.uint8)

        # get tumorous tissue
        tumorous_cell = cv2.bitwise_and(mask_tissue, filtered_subtract_img)
        tumorous_cell_path = f"{base_path}/{test_num}/step/13_tumor_cell.png"
        cv2.imwrite(tumorous_cell_path, tumorous_cell)
        
        # filter noise (might not need this step)
        filtered_tumor_cell = filter_small_components(tumorous_cell, min_component_size)
        filtered_tumor_cell_path = f"{base_path}/{test_num}/step/14_filtered_tumor_cell.png"
        cv2.imwrite(filtered_tumor_cell_path, filtered_tumor_cell)
        
        # * ADD TEMP FILE CALLER (FOR ORIGINAL IMAGE)
        
        # original/tumor/non-tumor area
        non_tumor_area = inverted_large_mask
        tumor_area = filtered_tumor_cell
        
        dcgen_img_path = f"{DCGEN_PATH}/{test_num}/{tiff_name}"
        dcgen_test_img = Image.open(dcgen_img_path)
        
        # class 0 (non-tumor)
        non_tumor_area_np = np.array(non_tumor_area)
        _, get_inverted_binary = cv2.threshold(non_tumor_area_np, 128, 1, cv2.THRESH_BINARY)
        tumor_inverted_binary = np.expand_dims(get_inverted_binary, axis=2)
        non_tumor_np = np.array(dcgen_test_img) * tumor_inverted_binary
        
        if non_tumor_np.shape[2] == 3:
            non_tumor_np[np.all(non_tumor_np == [0, 0, 0], axis=-1)] = [255, 255, 255]
        elif non_tumor_np.shape[2] == 4:
            non_tumor_np[np.all(non_tumor_np == [0, 0, 0, 0], axis=-1)] = [255, 255, 255, 255]
        
        non_tumor_img = Image.fromarray(non_tumor_np).convert("RGB")
        non_tumor_path= f"{base_path}/{test_num}/step/15_non_tumor.png"
        non_tumor_img.save(non_tumor_path)
        
        # class 1 (tumor)
        tumor_area_np = np.array(tumor_area)
        _, get_binary = cv2.threshold(tumor_area_np, 128, 1, cv2.THRESH_BINARY)
        tumor_binary = np.expand_dims(get_binary, axis=2)
        tumor_np = np.array(dcgen_test_img) * tumor_binary
        
        if tumor_np.shape[2] == 3:
            tumor_np[np.all(tumor_np == [0, 0, 0], axis=-1)] = [255, 255, 255]
        elif tumor_np.shape[2] == 4:
            tumor_np[np.all(tumor_np == [0, 0, 0, 0], axis=-1)] = [255, 255, 255, 255]
        
        tumor_np = Image.fromarray(tumor_np).convert("RGB")
        tumor_path= f"{base_path}/{test_num}/step/15_tumor.png"
        tumor_np.save(tumor_path)
        

def generate_patches(base_path, patch_size, information_threshold=0.4, label=1):
    print('-' * 25 + "  Start Generating Patches  " + '-' * 25)
    
    file_list = sorted(os.listdir(base_path))
    
    for file in tqdm(file_list, desc="Generating Patches"):
        file_name = file
        print('-' * 50)
        print(f"File Name: {file_name}") # e.g test1
        os.makedirs(f"{base_path}/{file_name}/patch/0", exist_ok=True)
        os.makedirs(f"{base_path}/{file_name}/patch/1", exist_ok=True)
        print('-' * 25 + "  Patch Directory Created  " + '-' * 25)
        
        if label == 1:
            data_path = f"{base_path}/{file_name}/step/15_tumor.png"
        elif label == 0:
            data_path = f"{base_path}/{file_name}/step/15_non_tumor.png"
        else:
            break
        
        slide = Image.open(data_path)
        slide_width, slide_height = slide.size
        print(slide.size)
        
        # calculate number of patches
        patch_num_width = slide_width // patch_size
        patch_num_height = slide_height // patch_size
        
        # extracts each patch
        for i in tqdm(range(patch_num_width), desc="Validating Cell Region"):
            for j in range(patch_num_height):
                x = i * patch_size
                y = j * patch_size
                patch = slide.crop((x, y, x+patch_size, y+patch_size))
                
                # get number of white pixels per patch
                patch_arr = np.array(patch)
                white_pixels = np.sum(np.all(patch_arr == [255, 255, 255], axis=-1))
                gray_pixels = np.sum(np.all(patch_arr == [224, 222, 223], axis=-1))
                total_pixels = patch_size * patch_size
                white_ratio = white_pixels / total_pixels
                gray_ratio = gray_pixels / total_pixels
                
                # filter no-info region
                if white_ratio + gray_ratio < information_threshold:
                    patch.save(f'{base_path}/{file_name}/patch/{label}/{file_name}_x{x}_y{y}_class{label}.png')
        
        patches = np.empty((patch_num_height, patch_num_width), dtype=object)

        # loads each patch and stores it in the array
        for i in range(patch_num_width):
            for j in range(patch_num_height):
                x = i * patch_size
                y = j * patch_size
                patch_ = os.path.join(f'{base_path}/{file_name}/patch/{label}/{file_name}_x{x}_y{y}_class{label}.png')
                
                if not os.path.exists(patch_):
                    patch = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
                else:
                    patch = Image.open(patch_)
                    patch = patch.convert('RGB')
                    
                patches[j][i] = patch

        # get path mask
        rows = []
        
        for i in range(patch_num_height):
            row = np.concatenate(patches[i], axis=1)
            rows.append(row)
            
        merged = np.concatenate(rows, axis=0)
        black_pixels = np.all(merged == [0, 0, 0], axis=-1)
        merged[black_pixels] = [255, 255, 255]

        # Saves the merged image
        merged_image = Image.fromarray(merged)
        merged_image.save(f"{base_path}/{file_name}/step/16_patch_mask_{label}.png")

 
if __name__ == "__main__":
    
    # raw_path: color preprocessed image
    # base_path: save checkpoint images
    # * ADD TIME FUNCTION
    # * patch size: 224×224, 512x512, 256x256
    # * patch size highly depends on the maginififcation factor
    
    raw_path = '../preprocessed_data' # ! check. 'killed' after two images
    base_path = '../preprocess_results/batch_5' # ! update
    
    # image_binarization(raw_path, base_path)
    
    generate_patches(base_path, patch_size=224, information_threshold=0.4, label=0)
    generate_patches(base_path, patch_size=224, information_threshold=0.4, label=1)
    
    