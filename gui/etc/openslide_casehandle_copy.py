import tifffile as tiff
import os, sys, shutil
from PIL import Image, ImageOps
import csv
from io import BytesIO
import tempfile
import time

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
import pyvips


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
    thumbnail_buffer = BytesIO()
    thumbnail_img.save(thumbnail_buffer, format="TIFF")
    thumbnail_image = pyvips.Image.new_from_buffer(thumbnail_img, "", access="sequential")

    # Save the thumbnail image to a temporary pyramidal TIFF file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
        temp_file_path = temp_file.name
        # Convert to a pyramidal TIFF with desired options
        thumbnail_image.tiffsave(temp_file_path,
                                compression="jpeg",
                                Q=90,
                                tile=True,
                                tile_width=256,
                                tile_height=256,
                                pyramid=True)

    # Load the converted pyramidal TIFF back as a pyvips image
    converted_image = pyvips.Image.new_from_file(temp_file_path)
    end_time_2 = time.time()
    duration_2 = end_time_1 - start_time_1
    print(f"##### Time for Getting Converting Image: {duration_1:.6f} secs")
    os.remove(temp_file_path)

    print("##### Good for now")
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