import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# import pyvips
# import openslide
import re
import os, sys
import numpy as np
import cv2
import time
from collections import OrderedDict
import torch

from PIL import Image, ImageOps, ImageDraw, ImageFont
import logging
import platform

OPENSLIDE_PATH = "C:/Users/junho/openslide-win64/bin"
VIPS_PATH = "C:/Users/junho/vips/bin"
POPPLER_PATH = "C:/Users/junho/poppler-23.07.0/Library/bin"

# install openslide for windows
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

os.environ["PATH"] = VIPS_PATH + os.pathsep + POPPLER_PATH + os.pathsep + os.environ["PATH"]
import pyvips
# os.environ["PATH"] = VIPS_PATH + os.pathsep + os.environ["PATH"]
import subprocess

# import poppler
# os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ["PATH"]
import pdf2image
from pdf2image import convert_from_path


Image.MAX_IMAGE_PIXELS = None
Image.MAX_MEMORY_USAGE = 0

from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.models import swin_v2_t, Swin_V2_T_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
import albumentations
from albumentations.pytorch import transforms
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

from torch.multiprocessing import Pool
# from multiprocessing import Pool


'''
generate patch
'''
def make_patch(pyramid_path, patch_size, mask_level, tissue_threshold):
    slide = openslide.OpenSlide(pyramid_path)
    slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[mask_level]))

    tissue_mask = slide.read_region((0,0), mask_level, slide.level_dimensions[mask_level])
    tissue_mask = cv2.cvtColor(np.array(tissue_mask), cv2.COLOR_RGBA2RGB)
    tissue_mask = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2HSV)
    tissue_mask = tissue_mask[:, :, 1]
    _, tissue_mask = cv2.threshold(tissue_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    tissue_mask = np.array(tissue_mask)
    
    p_size = patch_size
    width, height = np.array(slide.level_dimensions[0]) // p_size
    total = width * height
    all_cnt, patch_cnt = 0,0
    step = int(p_size / (2**mask_level))

    patch_list = []
    patch_location = []
    for i in range(width):
        for j in range(height):
            tissue_mask_sum = tissue_mask[step * j : step * (j+1),
                                          step * i : step * (i+1)].sum()
            tissue_mask_max = step * step * 255
            tissue_area_ratio = tissue_mask_sum / tissue_mask_max

            if tissue_area_ratio > tissue_threshold:
                patch = np.array(slide.read_region((p_size*i, p_size*j),0,(p_size,p_size)))
                patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
                patch_list.append(patch)
                patch_location += [[i,j]]
                patch_cnt += 1
            
            all_cnt +=1

    patch_list = np.array(patch_list)
    patch_location = np.array(patch_location)
    
    return patch_list, patch_location, width, height, slide_map, step

'''
import model
'''
def set_device(model, gpu_num):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # print(f"GPU DETECTED.")
        model = model.to(device)
    else:
        print(f"GPU NOT DETECTED.")
        sys.exit()
    return device, model

def prepare_model(use_model, checkpoint_model, num_classes):
    if use_model == 'Swin_v2_t':
        num_features = checkpoint_model.head.in_features
        checkpoint_model.head = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),
        
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.5),
        
        nn.Linear(128, num_classes)
    )
    elif use_model == 'Efficient_v2_s':
        num_features = checkpoint_model.classifier[1].in_features
        checkpoint_model.classifier[1] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            
            nn.Linear(128, num_classes)
        )
        
    return checkpoint_model

def update_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


'''
import data
'''
class dataset_eval(data.Dataset): 
    def __init__(self, patch_list, patch_location, transform=None):
        self.transform = transform
        self.location = patch_location
        self.data = patch_list

    def __getitem__(self,index):
        img = self.data[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, self.location[index]

    def __len__(self):
        return len(self.data)

def create_test_dataloader(patch_list, patch_location, batch_size, num_workers):
    albumentations_valid_transform = albumentations.Compose([
        albumentations.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensorV2(transpose_mask=True)
    ])
    test_dataset = dataset_eval(patch_list, patch_location, albumentations_valid_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader

'''
pred tumor
'''
def evaluate_model(threshold, model, test_loader, slide_map, step):
    tumor_data = []
    with torch.no_grad():
        for batch_idx, (inputs, location) in enumerate(test_loader):
            inputs = inputs.cuda()
            outputs = model(inputs)
            prob = F.softmax(outputs, 1, _stacklevel=5)
            pred = (prob[:, 1] >= threshold).long()

            for idx, output in enumerate(list(pred.data)):
                data = {"location": list(location[idx]), "prob": prob[idx, 1].item()}
                if output == 1:
                    tumor_data.append(data)

    tumor_location = np.array([data["location"] for data in tumor_data]).astype(int)
    pred_tumor_mask = np.zeros_like(slide_map)
    for h, w in tumor_location:
        cv2.rectangle(
            pred_tumor_mask,
            (step * h, step * w),
            (step * (h + 1), step * (w + 1)),
            (255, 255, 255),
            -1,
        )
    return pred_tumor_mask


'''
postprocess
'''
def cut_components(binary_mask, num_area, min_size):
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask)
    sizes = stats[:, -1]

    desc_sizes = sorted(range(nb_blobs), key=lambda x: sizes[x], reverse=True)

    im_result = np.zeros_like(im_with_separated_blobs, dtype=np.uint8)

    count = 0
    idx = 0
    while count < num_area and idx < len(desc_sizes):
        if sizes[desc_sizes[idx]] < min_size:
            break
        if int(binary_mask[im_with_separated_blobs == desc_sizes[idx]].mean()) == 255:
            im_result[im_with_separated_blobs == desc_sizes[idx]] = 255
            count += 1
        idx += 1

    return im_result

def filter_components(im_result, min_size):
    filtered_result = im_result.copy()
    _, labels, stats, _ = cv2.connectedComponentsWithStats(im_result)
    for label, stat in enumerate(stats):
        if stat[cv2.CC_STAT_AREA] < min_size:
            filtered_result[labels == label] = 0

    return filtered_result

def resize_and_save(image, width, height, save_path):
    resized_img = cv2.resize(image, (width, height))
    cv2.imwrite(save_path, resized_img)
    return resized_img

def process_and_save(image, width, height, tmp_path, step_name, dilate_size=35, gaussian_std=10, threshold_value=128):
    resized_img = resize_and_save(image, width, height, f"{tmp_path}/step3_{step_name}_resized.png")
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    resized_img = filter_components(resized_img.astype(np.uint8), ((patch_size * patch_size) * 2) / resized_factor)
    cv2.imwrite(f"{tmp_path}/step4_1_{step_name}_filtered_resized.png", resized_img)
    dilated_img = cv2.dilate(resized_img.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size,dilate_size)))
    cv2.imwrite(f"{tmp_path}/step4_2_{step_name}_dilated_resized.png", dilated_img)
    blurred_img = cv2.GaussianBlur(dilated_img, (0, 0), gaussian_std)
    cv2.imwrite(f"{tmp_path}/step5_{step_name}_blurred_resized.png", blurred_img)
    _, binary_blurred_img = cv2.threshold(blurred_img, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{tmp_path}/step6_{step_name}_binary_blurred_resized.png", binary_blurred_img)
    return binary_blurred_img

def find_contours_and_canvas(resized_image, slide_map):
    blank_canvas = np.zeros_like(slide_map, dtype=np.uint8)
    contours, _ = cv2.findContours(resized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return blank_canvas, contours

def process_contours_and_canvas(tmp_path, step, contours, blank_canvas, resized_slide_map):

    largest_area = 0
    infos = None

    for contour in contours:
        points = contour.reshape(-1, 2)
        x_min, y_min, x_max, y_max = points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        if area > largest_area:
            largest_area = area
            infos = [width, height]
        cv2.drawContours(blank_canvas, [contour], 0, (255, 255, 255), 30)

    resized_slide_map = cv2.bitwise_and(resized_slide_map, cv2.bitwise_not(blank_canvas))
    resized_slide_map = cv2.cvtColor(resized_slide_map, cv2.COLOR_BGR2RGB)

    cv2.imwrite(f"{tmp_path}/step8_{step}_final_contour.png", blank_canvas)
    cv2.imwrite(f"{tmp_path}/step9_{step}_apply_contour.png", resized_slide_map)
    return infos, resized_slide_map

def resize_and_draw(Y_MIN, Y_MAX, left, tmp_path, step, resized_slide_map, canvas, origin):
    ah, aw, _ = resized_slide_map.shape
    h, w, _ = canvas.shape
    dh, dw = h / ah, w / aw
    resized_canvas = cv2.resize(resized_slide_map, (w, h), interpolation=cv2.INTER_LINEAR)
    origin[Y_MIN:Y_MAX, left:left+w] = resized_canvas       # 이미지 복사
    cv2.imwrite(f"{tmp_path}/step10_{step}_draw_into_origin.png", origin)
    return dh, dw, origin


def draw_text(img, org, text, scale):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # font_path = "NanumSquareB.ttf"
    # font = ImageFont.truetype("NanumSquareB.ttf", scale)
    try:
        font = ImageFont.truetype(font_path, scale)
    except IOError:
        print(f"Error: Font file '{font_path}' not found or could not be opened.")
        return img
    
    draw.text(org, text, font=font, fill=(0, 0, 0))

    return np.array(img)

def get_info(PIXEL_WIDTH, PIXEL_HEIGHT, infos, dh, dw, X_MIN, Y_MAX):
    if infos is None:
        return 0, 0, 0
    else:
        _width, _height = infos

    _width, _height = infos
    _w = _width * dh * PIXEL_WIDTH
    _h = _height * dh * PIXEL_HEIGHT
    
    info_w, info_h = _w, _h
    info_area = _w * _h
    
    return info_w, info_h, info_area


def pipeline(data):
    start_time = time.time()
    slide_path = f"{input_path}/{data}"
    slide_name = slide_path.split('/')[-1].split('.')[0]
    
    '''
    tmp
    '''
    tmp_path = f"{tmp_dir}/{slide_name}"
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    thumbnail_input_path = f"{tmp_path}/step1_thumbnail.tif"
    thumbnail_output_path = f"{tmp_path}/step2_pyramid.tif"
    
    print(f"[{slide_name}] : Preprocess Start")
    
    '''
    get thumbnail
    '''
    raw_slide = Image.open(slide_path).convert("RGB")
    inverted_img = ImageOps.invert(raw_slide)
    thumbnail_coord = inverted_img.getbbox()
    thumbnail_img = raw_slide.crop(thumbnail_coord)
    thumbnail_img.save(thumbnail_input_path)
    
    '''
    convert pyramid
    '''
    
    pyramid_img = pyvips.Image.new_from_file(thumbnail_input_path)
    
    pyramid_img.tiffsave(thumbnail_output_path, compression="jpeg", tile=True,
            tile_width=256, tile_height=256, Q=90,
            pyramid=True)
    
    
    preprocess_end_time = time.time()
    preprocess_elapsed_time = preprocess_end_time - start_time
    print(f"[{slide_name}] : Preprocess Done | {preprocess_elapsed_time:.2f} seconds")
    logging.info(f"##\t[{slide_name}] : Preprocess Done | {preprocess_elapsed_time:.2f} seconds")
        
    '''
    inference
    '''
    
    num_classes = 2
    use_model = "Swin_v2_t"
    checkpoint_model = swin_v2_t(weights=None)
    checkpoint_model = prepare_model(use_model, checkpoint_model, num_classes)

    gpu_num = "0"
    device, checkpoint_model = set_device(checkpoint_model, gpu_num)

    state_dict = torch.load(checkpoint_path)
    if checkpoint_path.split("/")[-1].startswith("p_"):
        state_dict = update_state_dict(state_dict)
    checkpoint_model.load_state_dict(state_dict)

    checkpoint_model.eval()
    print(f"[{slide_name}] : Generate Patch Start")
    patch_list, patch_location, width, height, slide_map, step = make_patch(thumbnail_output_path, patch_size, map_level, tissue_threshold)

    generate_patch_end_time = time.time()
    generate_patch_elapsed_time = generate_patch_end_time - preprocess_end_time
    print(f"[{slide_name}] : Generate Patch Done | {generate_patch_elapsed_time:.2f} seconds")
    logging.info(f"##\t[{slide_name}] : Generate Patch Done | {generate_patch_elapsed_time:.2f} seconds")
 
    batch_size=100
    num_workers=0
    test_loader = create_test_dataloader(patch_list, patch_location, batch_size, num_workers)

    print(f"[{slide_name}] : Tumor Prediction Start")
    soft_pred_tumor_mask = evaluate_model(0.5, checkpoint_model, test_loader, slide_map, step)
    moderate_pred_tumor_mask = evaluate_model(0.65, checkpoint_model, test_loader, slide_map, step)
    hard_pred_tumor_mask = evaluate_model(0.8, checkpoint_model, test_loader, slide_map, step)
    
    prediction_end_time = time.time()
    prediction_elapsed_time = prediction_end_time - generate_patch_end_time
    print(f"[{slide_name}] : Tumor Prediction Done | {prediction_elapsed_time:.2f} seconds")
    logging.info(f"##\t[{slide_name}] : Tumor Prediction Done | {prediction_elapsed_time:.2f} seconds")
            
    '''
    postprocess
    '''
    resized_width = int(slide_map.shape[1] / resized_factor)
    resized_height = int(slide_map.shape[0] / resized_factor)
    data_dict = {"Soft": {}, "Moderate": {}, "Hard": {}}
    
    
    if not os.path.exists(output_path):
        os.makedirs(f"{output_path}")
    
    for step in data_dict.keys():
        if step == "Soft":
            step_num = 1
            data_dict[step]["pred_tumor_mask"] = soft_pred_tumor_mask
        elif step == "Moderate":
            step_num = 2
            data_dict[step]["pred_tumor_mask"] = moderate_pred_tumor_mask
        elif step == "Hard":
            step_num = 3
            data_dict[step]["pred_tumor_mask"] = hard_pred_tumor_mask
            
        data_dict[step]["resized_slide_map"] = resize_and_save(image=slide_map, width=resized_width, height=resized_height, save_path=f"{tmp_path}/{step}_slide_map_resized.png")

        data_dict[step]["resized_tumor_mask_image"] = process_and_save(image=data_dict[step]["pred_tumor_mask"], width=resized_width, height=resized_height, tmp_path=tmp_path, step_name=f"{step}_tumor_mask", gaussian_std=10, threshold_value=128)

        binary_blurred_mask = data_dict[step]["resized_tumor_mask_image"]
        # binary_blurred_mask_gray = cv2.cvtColor(binary_blurred_mask, cv2.COLOR_BGR2GRAY)
        resized_filtered_image = cut_components(binary_mask=binary_blurred_mask,
                                                num_area=2,
                                                min_size=((patch_size * patch_size) * 3) / resized_factor)
        cv2.imwrite(f"{tmp_path}/step7_1_{step}_filtered_binary_blurred_mask_resized.png", resized_filtered_image)
        resized_dilated_image = cv2.dilate(resized_filtered_image.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)))
        cv2.imwrite(f"{tmp_path}/step7_2_{step}_dilated_binary_blurred_mask_resized.png", resized_dilated_image)

        blank_canvas, contours = find_contours_and_canvas(resized_dilated_image, data_dict[step]["resized_slide_map"])
        data_dict[step]["blank_canvas"] = blank_canvas
        data_dict[step]["contours"] = contours
    
        contours = data_dict[step]["contours"]
        blank_canvas = data_dict[step]["blank_canvas"]
        resized_slide_map = data_dict[step]["resized_slide_map"]
        infos, resized_slide_map = process_contours_and_canvas(tmp_path, step, contours, blank_canvas, resized_slide_map)
        
        raw_slide = Image.open(slide_path).convert("RGB")
        resized_raw_slide = raw_slide.resize((ACTUAL_WIDTH, ACTUAL_HEIGHT))
        resized_inverted_img = ImageOps.invert(resized_raw_slide)
        resized_coord = resized_inverted_img.getbbox()
        resized_thumbnail_img = resized_raw_slide.crop(resized_coord)
        X_MIN, Y_MIN, X_MAX, Y_MAX = resized_coord
        image_width = X_MAX - X_MIN                
        left = (ACTUAL_WIDTH - image_width) // 2   
        
        origin = np.full_like(resized_raw_slide,(255,255,255), dtype=np.uint8)    
        canvas = cv2.cvtColor(np.array(resized_thumbnail_img), cv2.COLOR_BGR2RGB)  

        dh, dw, origin = resize_and_draw(Y_MIN, Y_MAX, left, tmp_path, step, resized_slide_map, canvas, origin)
        
        HEIGHT, WIDTH, _ = origin.shape
        PIXEL_WIDTH = PLATE_WIDTH / WIDTH
        PIXEL_HEIGHT = PLATE_HEIGHT / HEIGHT
        
        cv2.rectangle(origin, (left, Y_MIN), (left + image_width, Y_MAX), (0, 0, 0), 2)
        cv2.rectangle(origin, (0, 0), (WIDTH - 1, HEIGHT - 1), (0, 0, 0), 6)
        
        FONT_SCALE, THICKNESS = 0.6, 1
        step_title = f'{step}'
        output = draw_text(origin, (20, HEIGHT - 70), step_title, 50)
        info_w, info_h, info_area = get_info(PIXEL_WIDTH, PIXEL_HEIGHT, infos, dh, dw, X_MIN, Y_MAX)
        cv2.imwrite(f"{output_path}/{slide_name}_step_{step_num}_{step}_w_{info_w:.2f}_h_{info_h:.2f}_area_{info_area:.2f}.png", output)
    
    save_end_time = time.time()
    save_elapsed_time = save_end_time - prediction_end_time
    print(f"[{slide_name}] Save Result Done | {save_elapsed_time:.2f} seconds")
    logging.info(f"##\t[{slide_name}] Save Result Done | {save_elapsed_time:.2f} seconds")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[{slide_name}] All Process Done | {elapsed_time:.2f} seconds")
    logging.info(f"##\t[{slide_name}] All Process Done | {elapsed_time:.2f} seconds")


def generate_report(n_row, n_col, output_path, use_model):
    result_list = sorted(os.listdir(f"{output_path}"))
    total_images = len(result_list)
    images_per_page = n_row * n_col
    total_pages = (total_images) // images_per_page
    if total_images % images_per_page != 0:
        total_pages += 1

    pdf_filename = f'{datetime.today().strftime("%y%m%d_%H%M%S")}_Report.pdf'
    if not os.path.exists(report_path):
        os.makedirs(report_path)
    c = canvas.Canvas(f"{report_path}/{pdf_filename}", pagesize=A4)
    
    num = 1
    for page in range(total_pages):
        if page > 0:
            c.showPage()

        a4 = np.full((A4_PIXEL_HEIGHT, A4_PIXEL_WIDTH, 3), (255, 255, 255), dtype=np.uint8)
        if page == 0:
            raw_blank = ACTUAL_TOP
        else:
            raw_blank = ACTUAL_TOP_NEXT

        img_index = page * images_per_page

        if page == 0:
            title = 'Pathology AI - Report'
            a4 = draw_text(a4, ((A4_PIXEL_WIDTH - 1500) // 2, 150), title, 5 * 30)
            
            date = f'Date: {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}'
            a4 = draw_text(a4, (A4_PIXEL_WIDTH - 680, 350), date, 3 * 13)

            count = f'Total Slides : {int(total_images / 3)}'
            a4 = draw_text(a4, (A4_PIXEL_WIDTH - 680, 400), count, 3 * 13)

        
        for row in range(n_row):
            col_list = []
            for col in range(n_col):
                if img_index >= total_images:
                    break

                img = cv2.imread(f"{output_path}/{result_list[img_index]}")
                col_list.append(result_list[img_index])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                a4[
                    raw_blank:raw_blank + ACTUAL_HEIGHT,
                    ACTUAL_INTERVAL * (col + 1) + ACTUAL_WIDTH * col:ACTUAL_INTERVAL * (col + 1) + ACTUAL_WIDTH * (col + 1)
                ] = img
                img_index += 1
                
                if col == n_col - 1:
                    soft_data =  col_list[0]
                    soft_tumor_width = re.search(r"w_([\d.]+)", soft_data).group(1)
                    soft_tumor_height = re.search(r"h_([\d.]+)", soft_data).group(1)
                    soft_tumor_area = re.search(r"area_([\d.]+)", soft_data).group(1)
                    
                    moderate_data = col_list[1]
                    moderate_tumor_width = re.search(r"w_([\d.]+)", moderate_data).group(1)
                    moderate_tumor_height = re.search(r"h_([\d.]+)", moderate_data).group(1)
                    moderate_tumor_area = re.search(r"area_([\d.]+)", moderate_data).group(1)
                    
                    hard_data = col_list[2]
                    hard_tumor_width = re.search(r"w_([\d.]+)", hard_data).group(1)
                    hard_tumor_height = re.search(r"h_([\d.]+)", hard_data).group(1)
                    hard_tumor_area = re.search(r"area_([\d.]+)", hard_data).group(1)
                    
                    extract_slide_name = re.search(r"(.*?)_step", soft_data)
                    if len(extract_slide_name.group(1)) > 25:
                        slide_name = extract_slide_name.group(1)[:22] + '...'
                    else:
                        slide_name = extract_slide_name.group(1)
                
                    slide_info = f"Slide Name : [{num}]\n{slide_name}\n\nTumor Information :\n[Biggest Tumor]\n\n"
                    
                    slide_info += f"Soft\n- Size : {soft_tumor_width} x {soft_tumor_height} mm\n- Area : {soft_tumor_area} mm²\n\n"
                    slide_info += f"Moderate\n Size : {moderate_tumor_width} x {moderate_tumor_height} mm\n- Area : {moderate_tumor_area} mm²\n\n"
                    slide_info += f"Hard\n- Size : {hard_tumor_width} x {hard_tumor_height} mm\n- Area : {hard_tumor_area} mm²\n\n"

                    top_left = (ACTUAL_TEXT_WIDTH + 700, raw_blank)
                    bottom_right = (ACTUAL_TEXT_WIDTH + 1400, raw_blank + ACTUAL_TEXT_HEIGHT)
                    # cv2.rectangle(a4, top_left, bottom_right, (0, 0, 0), thickness=4)
                    
                    text_start = (top_left[0] + 10, top_left[1] + 10)
                    a4 = draw_text(a4, text_start, slide_info, 3 * 15)
                    
                    num += 1

            raw_blank += ACTUAL_HEIGHT + ACTUAL_MIDDLE

        page_num = str(page + 1)
        a4 = draw_text(a4, (A4_PIXEL_WIDTH - 100, A4_PIXEL_HEIGHT - 100), page_num, 40)

        a4_pil = Image.fromarray(a4)
        c.drawInlineImage(a4_pil, 0, 0, width=A4[0], height=A4[1])

    c.save()


def print_gpu_info():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)  # Get the name of the first GPU device
        print(f"GPU: {device}")
        logging.info(f"##\tGPU: {device}")
    else:
        print("##\tNo GPU/CUDA available.")

def print_os_info():
    system = platform.system()
    release = platform.release()
    print(f"Operating System: {system} {release}")
    logging.info(f"##\tOperating System: {system} {release}")

# if __name__ == "__main__":
'''
Report settings
'''
n_row = 3
n_col = 3

A4_PIXEL_WIDTH = 2480
A4_PIXEL_HEIGHT = 3508
A4_WIDTH = 210
A4_HEIGHT = 297
A4_WIDTH_RATIO = 11.80952381
A4_HEIGHT_RATIO = 11.811447811

top = 40
PLATE_WIDTH = 27 # 26 / 27 / 28
PLATE_HEIGHT = 78 # 75 / 78 / 80

w_interval = 15
h_interval = 5

ACTUAL_INTERVAL = int(w_interval * A4_WIDTH_RATIO)
ACTUAL_TOP = int(top * A4_HEIGHT_RATIO)
ACTUAL_TOP_NEXT = int(30 * A4_HEIGHT_RATIO)
ACTUAL_MIDDLE = int(h_interval * A4_HEIGHT_RATIO)
ACTUAL_WIDTH = int(PLATE_WIDTH * A4_WIDTH_RATIO)
ACTUAL_HEIGHT = int(PLATE_HEIGHT * A4_HEIGHT_RATIO)
ACTUAL_TEXT_WIDTH = int(80 * A4_WIDTH_RATIO)
ACTUAL_TEXT_HEIGHT = int(PLATE_HEIGHT * A4_HEIGHT_RATIO)

'''
paramter
'''
resized_factor = 8
map_level = 0
tissue_threshold = 0.0

patch_size = 192
use_model = "Swin_v2_t"
# checkpoint_path = '/mnt/tesser_nas2/AI_DataSets/DCGEN/Train/models/p192_swin_v2_t/fold0/p_model_best_accuracy_0.9548.pth'
# checkpoint_path = '/mnt/tesser_nas2/AI_DataSets/DCGEN/Train/models/p192_efficient_v2_s/fold0/s_model_best_accuracy_0.9582.pth'
checkpoint_path = './checkpoint/p_model_best_accuracy_0.9548.pth'

'''
config
'''
origin_path = './prediction'
input_path = f'{origin_path}/input'
output_path = f'{origin_path}/output'
report_path = f"{origin_path}/report"
tmp_dir = f"{origin_path}/tmp"

log_path = f"{origin_path}/log"
font_path = "./design/NanumSquareB.ttf"
    
if __name__ == "__main__":
    if not os.path.exists(f"{log_path}"):
        os.makedirs(f"{log_path}")
    log_file = os.path.join(log_path, "loggins_track.log")
    
    logging.basicConfig(filename=f'{log_file}', level=logging.INFO, format='%(message)s', filemode='a')
    print("##\t0. Starting Pipeline...")
    logging.info("\n================================================================")
    logging.info("##\t0. Starting Pipeline...")
    print_gpu_info()
    print_os_info()

    '''
    multiprocess
    '''
    total_start_time = time.time()
    workers = 1
    slide_list = sorted(os.listdir(input_path))
    chunks = [slide_list[i:i + workers] for i in range(0, len(slide_list), workers)]
    print(f"All Series : \n{chunks}")
    logging.info(f"##\tAll Series : \n{chunks}")
    logging.info(f"##\tNum Workers : {workers}")

    with Pool(processes=workers) as pool:
        for chunk in chunks:
            print(f"Sub series : {chunk}")
            logging.info(f"##\tSub series : {chunk}")
            print(f"Processing {len(chunk)} images in Sub series.")
            logging.info(f"##\tProcessing {len(chunk)} images in Sub series.")

            start_time = time.time()
            pool.map(pipeline, chunk)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Sub Series process time | {elapsed_time:.2f} seconds")
            logging.info(f"##\tSub Series process time | {elapsed_time:.2f} seconds")
            
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"All series process time | {total_elapsed_time:.2f} seconds")
    logging.info(f"##\tAll series process time | {total_elapsed_time:.2f} seconds")
    
    '''
    generate report
    '''
    generate_report(n_row=n_row, n_col=n_col, output_path=output_path, use_model=use_model)
