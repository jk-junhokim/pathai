import tifffile as tiff
import os, sys, shutil
from PIL import Image, ImageOps
import csv

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
slide_save_name = "sample.tif"

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
    thumbnail_coord = inverted_img.getbbox()
    thumbnail_img = raw_image.crop(thumbnail_coord)
    thumbnail_img.save(f"{thumbnail_save_path}/{slide_save_name}")

    # save csv for mask creation
    with open(f"{csv_save_path}/metadata.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["slide_name", "Thumbnail_coord"])
            writer.writerow([slide_save_name, thumbnail_coord])

    input_path = f"{thumbnail_save_path}/{slide_save_name}"
    output_path = f"{pyramid_save_path}/{slide_save_name}"

    command = [
        "vips",
        "tiffsave",
        input_path,
        output_path,
        "--compression=jpeg",
        "--Q=90",
        "--tile",
        "--tile-width=256",
        "--tile-height=256",
        "--pyramid",
        "--vips-leak",
    ]

    try:
        subprocess.run(command, check=True)
        print("##### Converting Image")
    except subprocess.CalledProcessError as e:
        print("##### Error:", e)

    try:
        slide = openslide.OpenSlide(output_path)
    except openslide.OpenSlideError:
        print(f"Error: Cannot Open OpenSlide Module. {output_path}")
        sys.exit()
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}\n\n")