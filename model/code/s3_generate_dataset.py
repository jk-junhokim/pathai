import openslide
import cv2
import sys
import numpy as np
import os
import random
import shutil
import logging

from xml.etree.ElementTree import parse
from PIL import Image
from multiprocessing import Process
import my_arguments
from my_arguments import PreprocessArgs
from s1_param import MyHyperparmeter as hp


def make_dir(args, slide_name, flags):
    """
    Make directory of files using flags.
    If flags is tumor_patch or normal patch additional directory
    handling is needed.

    Args:
        slide_name (str): name of slide used
        flags (str): various flags are existed below

    Returns:
        (str): directory path to flag creation
    """
    if flags == "slide":
        return f"{args.slide_path}/{slide_name}.tif"

    elif flags == "xml":
        return f"{args.xml_path}/{slide_name}.xml"

    elif flags == "mask":
        return f"{args.mask_path}"

    elif flags == "map":
        if not os.path.exists(f"{args.mask_path}/b_{slide_name}"):
            os.makedirs(f"{args.mask_path}/b_{slide_name}")
        return f"{args.mask_path}/b_{slide_name}/b_{slide_name}_map.png"

    elif flags == "tumor_mask_temp":
        if not os.path.exists(f"{args.mask_path}/b_{slide_name}"):
            os.makedirs(f"{args.mask_path}/b_{slide_name}")
        return f"{args.mask_path}/b_{slide_name}/b_{slide_name}_tumor_mask_temp.png"
    
    elif flags == "tumor_mask":
        if not os.path.exists(f"{args.mask_path}/b_{slide_name}"):
            os.makedirs(f"{args.mask_path}/b_{slide_name}")
        return f"{args.mask_path}/b_{slide_name}/b_{slide_name}_tumor_mask.png"

    elif flags == "normal_mask":
        if not os.path.exists(f"{args.mask_path}/b_{slide_name}"):
            os.makedirs(f"{args.mask_path}/b_{slide_name}")
        return f"{args.mask_path}/b_{slide_name}/b_{slide_name}_normal_mask.png"

    elif flags == "tissue_mask":
        if not os.path.exists(f"{args.mask_path}/b_{slide_name}"):
            os.makedirs(f"{args.mask_path}/b_{slide_name}")
        return f"{args.mask_path}/b_{slide_name}/b_{slide_name}_tissue_mask.png"

    elif flags == "tumor_patch":
        if not os.path.exists(f"{args.patch_path}/b_{slide_name}/tumor"):
            os.makedirs(f"{args.patch_path}/b_{slide_name}/tumor")
        return f"{args.patch_path}/b_{slide_name}/tumor"

    elif flags == "normal_patch":
        if not os.path.exists(f"{args.patch_path}/b_{slide_name}/normal"):
            os.makedirs(f"{args.patch_path}/b_{slide_name}/normal")
        return f"{args.patch_path}/b_{slide_name}/normal"

    else:
        print("make_dir flags error")
        return

def check_file(args, filedir, filename):
    """
    Check if file(filename) exists in filedir.

    Args:
        fliedir (str): directory of the file
        filename (str): name of the file
        
    Returns:
        (bool): true if file exists
    """
    exist = False

    os.chdir(filedir) # change working directory
    cwd = os.getcwd() # assign directory

    for file_name in os.listdir(cwd):
        if file_name == filename:
            exist = True

    os.chdir(args.origin_path) # return to original working directory
    
    return exist

def read_xml(args, slide_name, mask_level):
    """
    Read xml files with tumor coordinates.

    Args:
        slide_name (str): name of slide
        mask_level (int): level of mask

    Returns:
        (numpy array): coordinates of tumor areas
    """
    path = make_dir(args, slide_name, "xml")
    xml = parse(path).getroot()
    coors_list = []
    coors = []
    
    for areas in xml.iter("Coordinates"):
        for area in areas:
            coors.append(
                [
                    round(float(area.get("X")) / (2**mask_level)),
                    round(float(area.get("Y")) / (2**mask_level)),
                ]
            )
        coors_list.append(coors)
        coors = []
        
    if len(coors_list) == 1:
        return np.array(coors_list)
    
    else:
        return np.array(coors_list, dtype=list)

def make_mask(args, slide_name, mask_level):
    """
    Make tumor, normal, tissue mask using xml files
    and otsu threshold (ASAP).

    Args:
        slide_name (str): name of slide
        mask_level (int): level of mask
    """
    # get paths to directories
    slide_path = make_dir(args, slide_name, "slide")
    mask_folder_path = make_dir(args, slide_name, "mask")
    map_path = make_dir(args, slide_name, "map")
    tumor_mask_path_temp = make_dir(args, slide_name, "tumor_mask_temp")
    tumor_mask_path = make_dir(args, slide_name, "tumor_mask")
    normal_mask_path = make_dir(args, slide_name, "normal_mask")
    tissue_mask_path = make_dir(args, slide_name, "tissue_mask")
    
    # slide loading
    try:
        slide = openslide.OpenSlide(slide_path)
    except openslide.OpenSlideError:
        print(f"###########\tError: Cannot Open OpenSlide Module. {slide_path}")
        logging.info(f"###########\tError: Cannot Open OpenSlide Module. {slide_path}")
        return
    
    slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[args.map_level]))
    coors_list = read_xml(args, slide_name, mask_level)
 
    
    tissue_mask = np.zeros((slide.level_dimensions[mask_level][1], slide.level_dimensions[mask_level][0]), dtype=np.uint8)
    tumor_mask = np.zeros((slide.level_dimensions[mask_level][1], slide.level_dimensions[mask_level][0]), dtype=np.uint8)

    tumor_mask_exist = check_file(args, f"{mask_folder_path}/b_{slide_name}", f"b_{slide_name}_tumor_mask.png")

    if tumor_mask_exist == False:
        for coors in coors_list:
            try:
                cv2.drawContours(slide_map, np.array([coors]), -1, 255, 1)
                cv2.drawContours(tumor_mask, np.array([coors]), -1, 255, -1)
            except cv2.error as e:
                print(f"###########\tError occurred while drawing contours: {e}")
                logging.info(f'###########\tError while creating contours: {e}')
                return
                
    cv2.imwrite(tumor_mask_path_temp, tumor_mask)        
    cv2.imwrite(map_path, slide_map)
    
    # check tumor mask / draw tumor mask
    tumor_mask_exist = check_file(args, f"{mask_folder_path}/b_{slide_name}", f"b_{slide_name}_tumor_mask.png")

    # check tissue mask / draw tissue mask
    tissue_mask_exist = check_file(args, f"{mask_folder_path}/b_{slide_name}", f"b_{slide_name}_tissue_mask.png")
    
    if tissue_mask_exist == False:
        slide_lv = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
        slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
        slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
        slide_lv = slide_lv[:, :, 1]
        _, tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(tissue_mask_path, np.array(tissue_mask))
        
    tumor_tissue_area = cv2.bitwise_and(tissue_mask, tumor_mask)
    cv2.imwrite(tumor_mask_path, tumor_tissue_area)
        
    # check normal mask / draw normal mask
    normal_mask_exist = check_file(args, f"{mask_folder_path}/b_{slide_name}", f"b_{slide_name}_normal_mask.png")
    
    if normal_mask_exist == False:
        tumor_mask = cv2.imread(tumor_mask_path_temp, 0)
        tissue_mask = cv2.imread(tissue_mask_path, 0)
        height, width = np.array(tumor_mask).shape
        
        for i in range(width):
            for j in range(height):
                if tumor_mask[j][i] > 127:
                    tissue_mask[j][i] = 0
                    
        normal_mask = np.array(tissue_mask)
        cv2.imwrite(normal_mask_path, normal_mask)
    
def make_patch(args, slide_name, mask_level):
    """
    Extract normal, tumor patches using normal, tumor mask.

    Args:
        slide_num (int): number of slide used
        mask_level (int): level of mask
    """
    slide_path = make_dir(args, slide_name, "slide")
    map_path = make_dir(args, slide_name, "map")
    mask_folder_path = make_dir(args, slide_name, "mask")
    tumor_mask_path = make_dir(args, slide_name, "tumor_mask")
    tumor_patch_path = make_dir(args, slide_name, "tumor_patch")
    normal_mask_path = make_dir(args, slide_name, "normal_mask")
    normal_patch_path = make_dir(args, slide_name, "normal_patch")

    # parameter for sensitivity
    tumor_threshold = args.tumor_threshold
    # tumor_sel_max = args.tumor_sel_max
    normal_threshold = args.normal_threshold
    # normal_sel_max = args.normal_sel_max
    
    tumor_mask_exist = check_file(args, f"{mask_folder_path}/b_{slide_name}", f"b_{slide_name}_tumor_mask.png")
    normal_mask_exist = check_file(args, f"{mask_folder_path}/b_{slide_name}", f"b_{slide_name}_normal_mask.png")
    
    if (tumor_mask_exist and normal_mask_exist) == False:
        print('###########\tTumor or Normal Mask does NOT EXIST...')
        logging.info('###########\tTumor or Normal Mask does NOT EXIST...')
        return
    
    slide = openslide.OpenSlide(slide_path)
    slide_map = cv2.imread(map_path, -1)
    tumor_mask = cv2.imread(tumor_mask_path, 0)
    normal_mask = cv2.imread(normal_mask_path, 0)
    
    patch_size = args.patch_size
    width, height = np.array(slide.level_dimensions[0]) // patch_size
    real_width, real_height = np.array(slide.level_dimensions[0])
    total = width * height # total of patches that can be generated from whole slide
    num_tumor_patch = 0
    num_normal_patch = 0
    tumor_limit = False
    normal_limit = False
    stride = int(patch_size / (2**mask_level))
    args.slide_tracker += 1
    args.width_tracker += real_width
    args.height_tracker += real_height
    w_avg = int(args.width_tracker / args.slide_tracker)
    h_avg = int(args.height_tracker / args.slide_tracker)
    print(f"width : {real_width} | height : {real_height} | total : {total} | patch_size : {patch_size}")
    print(f"Slide # Count: {args.slide_tracker} | Width Avg: {w_avg} | Height Avg: {h_avg}")
    logging.info(f"width : {real_width} | height : {real_height} | total : {total} | patch_size : {patch_size}")
    logging.info(f"Slide # Count: {args.slide_tracker} | Width Avg: {w_avg} | Height Avg: {h_avg}")
    
    for i in range(width):
        for j in range(height):
            tumor_mask_sum = tumor_mask[
                stride * j : stride * (j + 1), stride * i : stride * (i + 1)
            ].sum()
            normal_mask_sum = normal_mask[
                stride * j : stride * (j + 1), stride * i : stride * (i + 1)
            ].sum()
            mask_max = stride * stride * 255
            tumor_area_ratio = tumor_mask_sum / mask_max
            normal_area_ratio = normal_mask_sum / mask_max

            # extract tumor patch
            if ((tumor_area_ratio > tumor_threshold) and not tumor_limit):
                patch_name = f"{tumor_patch_path}/t_b_{slide_name}_x{patch_size * i}_y{patch_size * j}_class1.png"
                patch = slide.read_region((patch_size * i, patch_size * j), 0, (patch_size, patch_size))
                patch.save(patch_name)
                
                cv2.rectangle(
                    slide_map,
                    (stride * i, stride * j),
                    (stride * (i + 1), stride * (j + 1)),
                    (0, 0, 255),
                    3)
                
                num_tumor_patch += 1
                # tumor_limit = num_tumor_patch > tumor_sel_max
            # extract normal patch
            elif ((normal_area_ratio > normal_threshold) and (tumor_area_ratio == 0) and not normal_limit):
                patch_name = f"{normal_patch_path}/n_b_{slide_name}_x{patch_size * i}_y{patch_size * j}_class0.png"
                patch = slide.read_region((patch_size * i, patch_size * j), 0, (patch_size, patch_size))
                patch.save(patch_name)
                
                cv2.rectangle(
                    slide_map,
                    (stride * i, stride * j),
                    (stride * (i + 1), stride * (j + 1)),
                    (0, 0, 0),
                    3)
                
                num_normal_patch += 1
                # normal_limit = num_normal_patch > normal_sel_max
            else:
                pass

            # check max boundary of patch
            # if normal_limit and tumor_limit:
            #     print('###########\tReached Patch Extraction Limit...\n')
            #     logging.info('###########\tReached Patch Extraction Limit...\n')
            #     cv2.imwrite(map_path, slide_map)
            #     return
    
    progress_str = "All: %d, Normal: %d, Tumor: %d" % (
                num_normal_patch + num_tumor_patch,
                num_normal_patch,
                num_tumor_patch)
    print(progress_str)
    logging.info(progress_str)  
    cv2.imwrite(map_path, slide_map)

def merge_patch(args, slide_name):
    """
    Merges all patches into one file path.
    Both tumor and normal patches.

    Args:
        slide_name (str): name of slide
    """
    tumor_patch_path = make_dir(args, slide_name, "tumor_patch")
    normal_patch_path = make_dir(args, slide_name, "normal_patch")

    tumor_files = os.listdir(tumor_patch_path)
    tumor_num = len(tumor_files)
    random.shuffle(tumor_files)

    normal_files = os.listdir(normal_patch_path)
    normal_num = len(normal_files)
    random.shuffle(normal_files)

    os.chdir(tumor_patch_path)

    if not os.path.exists(f"{args.dataset_path}/merge/"):
        os.makedirs(f"{args.dataset_path}/merge/")
    for i in range(tumor_num):
        shutil.copy(tumor_files[i], f"{args.dataset_path}/merge/")

    os.chdir(normal_patch_path)

    for i in range(normal_num):
        shutil.copy(normal_files[i], f"{args.dataset_path}/merge/")
        
def split_patch(args):
    """
    Splits each patch type into train, validation, and test datasets.
    """
    all_patch_list = sorted(os.listdir(f"{args.dataset_path}/merge/"))
    random.shuffle(all_patch_list)
    normal_patch_list = []
    tumor_patch_list = []
    split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    data_type = ['0', '1'] # 0 = normal, 1 = tumor
    
    imgs = {}
    imgs['train'] = []
    imgs['val'] = []
    imgs['test'] = []
    
    train_files = set()
    val_files = set()
    test_files = set()
    
    for file_name in all_patch_list:
        if file_name.startswith("n_"):
            normal_patch_list.append(file_name)
        elif file_name.startswith("t_"):
            tumor_patch_list.append(file_name)
            
    normal_count = len(normal_patch_list)
    tumor_count = len(tumor_patch_list)
    print(f"Normal Patch # : {normal_count}")
    print(f"Tumor Patch # : {tumor_count}")
    logging.info(f"Normal Patch # : {normal_count}")
    logging.info(f"Tumor Patch # : {tumor_count}")
    
    random.shuffle(normal_patch_list)
    random.shuffle(tumor_patch_list)
    # patch_list = []
    # patch_list = [normal_patch_list, tumor_patch_list]
    
    balanced_list = []
    if normal_count >= tumor_count:
        random.shuffle(normal_patch_list)
        balanced_normal_patch_list = random.sample(normal_patch_list, tumor_count)
        balanced_tumor_patch_list = tumor_patch_list
        bn = len(balanced_normal_patch_list)
        bt = len(balanced_tumor_patch_list)
        if bn == bt:
            print(f"Patch Numbers are BALANCED at {bn}")
            logging.info(f"Patch Numbers are BALANCED at {bn}")
        else:
            print(f"Patches are NOT BALANCED")
            logging.info(f"Patches are NOT BALANCED")
    else:
        random.shuffle(tumor_patch_list)
        balanced_tumor_patch_list = random.sample(tumor_patch_list, normal_count)
        balanced_normal_patch_list = normal_patch_list
        bn = len(balanced_normal_patch_list)
        bt = len(balanced_tumor_patch_list)
        if bn == bt:
            print(f"Patch Numbers are BALANCED at {bn}")
            logging.info(f"Patch Numbers are BALANCED at {bn}")
        else:
            print(f"Patches are NOT BALANCED")
            logging.info(f"Patches are NOT BALANCED")
    
    balanced_list = [balanced_normal_patch_list, balanced_tumor_patch_list]
    
    for c in data_type:
        file_list = balanced_list[int(c)]
        total_files = len(file_list)
        train_end = int(total_files * split_ratios['train'])
        val_end = int(total_files * (split_ratios['train'] + split_ratios['val']))
        
        imgs['train'] = file_list[:train_end]
        imgs['val'] = file_list[train_end:val_end]
        imgs['test'] = file_list[val_end:]
        lt = len(imgs['train'])
        lv = len(imgs['val'])
        ltt = len(imgs['test'])
        
        train_files.update(imgs['train'])
        val_files.update(imgs['val'])
        test_files.update(imgs['test'])
        
        print("###########\tClass", c, "train:", len(imgs['train']), ", val:", len(imgs['val']), ", test:", len(imgs['test']))
        logging.info(f"###########\tClass: {c} | Train: {lt} |  Val: {lv} | Test: {ltt}")
        
        for set_type in ['train', 'val', 'test']:
            for img_name in imgs[set_type]:
                img = Image.open(f"{args.dataset_path}/merge/" + img_name)
                dest_path = os.path.join(f"{args.dataset_path}/", set_type)
                
                if not os.path.exists(dest_path):
                    os.mkdir(dest_path)
                dest_path = os.path.join(dest_path, os.path.basename(img_name))
                img.save(dest_path, "PNG")
    
        # N_train_total = len(imgs['train'])
        # N_val_total = len(imgs['val'])
        # N_test_total = len(imgs['test'])
        # print("###########\tNormal Patches:")
        # print("###########\tTrain Set: ",  N_train_total)
        # print("###########\tValidation Set: ",  N_val_total)
        # print("###########\tTest Set: ",  N_test_total)
        # print("###########\tTumor Patches:")
        # print("###########\tTrain Set: ",  T_train_total)
        # print("###########\tValidation Set: ",  T_val_total)
        # print("###########\tTest Set: ",  T_test_total)
        # print("###########\tData split successful")
        # print("###########\tChecking for overlapping files...")
    
        logging.info("###########\tData split successful")
        logging.info("###########\tChecking for overlapping files...")

        overlapping_files = train_files.intersection(val_files, test_files)
        of_len = len(overlapping_files)
        if len(overlapping_files) == 0:
            print("###########\tNo Overlap")
            logging.info("###########\tNo Overlap")
        else:
            print("###########\tOverlap Occured")
            print(f"###########\tNumber of Overlaps: {of_len}")
            logging.info("###########\tOverlap Occured")
            logging.info(f"###########\tNumber of Overlaps: {of_len}")


def generate_text_list(args):
    f = open(f"{args.dataset_path}/train.txt", "w")
    for patch in os.listdir(f"{args.dataset_path}/train/"):
        if patch.startswith("n_"):
            f.write(f"{args.dataset_path}/train/{patch} 0\n")
        elif patch.startswith("t_"):
            f.write(f"{args.dataset_path}/train/{patch} 1\n")
    f.close()
    
    f = open(f"{args.dataset_path}/val.txt", "w")
    for patch in os.listdir(f"{args.dataset_path}/val/"):
        if patch.startswith("n_"):
            f.write(f"{args.dataset_path}/val/{patch} 0\n")
        elif patch.startswith("t_"):
            f.write(f"{args.dataset_path}/val/{patch} 1\n")
    f.close()
    
    f = open(f"{args.dataset_path}/test.txt", "w")
    for patch in os.listdir(f"{args.dataset_path}/test/"):
        if patch.startswith("n_"):
            f.write(f"{args.dataset_path}/test/{patch} 0\n")
        elif patch.startswith("t_"):
            f.write(f"{args.dataset_path}/test/{patch} 1\n")
    f.close()


if __name__ == "__main__":
    args = PreprocessArgs().parse()
    slides_path = args.slide_path # tiff image directory
    if not os.path.exists(f"{args.dataset_path}"):
            os.makedirs(f"{args.dataset_path}")
    log_file = os.path.join(args.dataset_path, "loggins.log")
    
    logging.basicConfig(filename=f'{log_file}', level=logging.INFO, format='%(message)s', filemode='a')
    print("###########\t0. Start Preprocessing...")
    logging.info("###########\t0. Start Preprocessing...")
    
    max_limit = 50
    
    for index, slide_ in enumerate(sorted(os.listdir(slides_path))):
        if index < max_limit:        
            SLIDE = slide_.split(".")[0]
            MASK_LEVEL = args.map_level
            print(f"\nINDEX : {index} | SLIDE_NAME : {SLIDE}")
            logging.info(f"\nINDEX : {index} | SLIDE_NAME : {SLIDE}")
            
            make_mask(args, SLIDE, MASK_LEVEL)
            make_patch(args, SLIDE, MASK_LEVEL)
            merge_patch(args, SLIDE)
    
    print("\n###########\t4. Splitting Patch-Level Dataset...\n")
    logging.info("###########\t4. Splitting Patch-Level Dataset...\n")
    split_patch(args)
    
    print('###########\t5. Generating Text Data...')
    logging.info('###########\t5. Generating Text Data...')
    generate_text_list(args)
    
    print(f'###########\t6. Done: {hp.DATASET_NAME}_{hp.PATCH_SIZE}')
    logging.info(f'###########\t6. Done: {hp.DATASET_NAME}_{hp.PATCH_SIZE}')
    