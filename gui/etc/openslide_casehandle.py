import tifffile as tiff
import os, sys, shutil
from PIL import Image, ImageOps
import csv
from io import BytesIO
import tempfile
import time
import numpy as np
from PIL import Image, ImageQt
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QGraphicsPixmapItem
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic

# module path
OPENSLIDE_PATH = "C:/Users/junho/openslide-win64/bin"
VIPS_PATH = "C:/Users/junho/vips/bin"
# import openslide
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
# set up vips
os.environ["PATH"] = VIPS_PATH + os.pathsep + os.environ["PATH"]
import subprocess


Image.MAX_IMAGE_PIXELS = None
Image.MAX_MEMORY_USAGE = 0

# image path
raw_path = "./test_images/OF9003.tif"
thumbnail_save_path = "./sample_thumbnail"
csv_save_path = "./sample_csv"
pyramid_save_path = "./sample_pyramid"
slide_save_name = "sample_1.tif"

# read original image
if __name__ == "__main__" :
    input_image = raw_path
    raw_image = Image.open(input_image)
    if raw_image.mode == "RGBA":
        r_ch, g_ch, b_ch, alpha_ch = raw_image.split()
    elif raw_image.mode == "RGB":
        r_ch, g_ch, b_ch = raw_image.split()
    print("##### Cropping Image")
    inverted_img = ImageOps.invert(Image.merge("RGB", (r_ch, g_ch, b_ch)))
    start_time = time.time()
    thumbnail_coord = inverted_img.getbbox()
    end_time = time.time()
    duration = end_time - start_time
    print(f"##### Time for Getting Target Area: {duration:.4f}")
    start_time_1 = time.time()
    thumbnail_img = raw_image.crop(thumbnail_coord) 
    end_time_1 = time.time()
    duration_1 = end_time_1 - start_time_1
    print(f"##### Time for Cropping: {duration_1:.4f}")

    # Store the thumbnail_img in a temporary file
    start_time_2 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
                temp_file_path = temp_file.name
                thumbnail_img.save(temp_file_path, format="TIFF")

    command = [
        "vips",
        "tiffsave",
        temp_file_path,  # Use the temporary file path here
        "-",  # Use "-" to indicate writing to standard output (stdout)
        "--compression=jpeg",
        "--Q=90",
        "--tile",
        "--tile-width=256",
        "--tile-height=256",
        "--pyramid",
        "--vips-leak",
    ]

    try:
        print("Converting Images")
        completed_process = subprocess.run(command, stdout=subprocess.PIPE, check=True)
        image_data = completed_process.stdout # 'image_data' contains the binary image data as a bytes object
    except subprocess.CalledProcessError as e:
        print("##### Error:", e)

    # Convert the bytes into a PIL Image
    # image_bytes_io = BytesIO(image_data)
    # pil_image = Image.open(image_bytes_io)
    # Convert PIL Image to QImage
    # qt_image = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGB888)
    buffer = QBuffer()
    buffer.open(QBuffer.WriteOnly)
    buffer.write(image_data)
    buffer.seek(0)
    qt_image = QImage.fromData(buffer.data())
    qt_pixmap = QPixmap.fromImage(qt_image)
    if not qt_pixmap.isNull():  # Check if the loaded image is valid
        print("##### Correct!")
        pixmap_item = QGraphicsPixmapItem(qt_pixmap)
        print("##### Correct!")
    else:
         print("##### Wrong!")


    print("##### Run Complete")
    # try:
    #     slide = openslide.OpenSlide(image_data)
    # except openslide.OpenSlideError:
    #     print(f"Error: Cannot Open OpenSlide Module. {image_data}")
    #     sys.exit()

    # try:
    #     slide = openslide.OpenSlide(output_path)
    # except openslide.OpenSlideError:
    #     print(f"Error: Cannot Open OpenSlide Module. {output_path}")
    #     sys.exit()

    # print(f"File id: {slide}")
    # print(f"Dimensions: {slide.dimensions}")
    # print(f"Number of levels in the image: {slide.level_count}")
    # print(f"Downsample factor per level: {slide.level_downsamples}")
    # print(f"Dimensions of levels: {slide.level_dimensions}\n\n")