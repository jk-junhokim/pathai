import os
import sys
import numpy as np
from PIL import Image

##################################################
############# Installation Guideline #############
##################################################
# 1. Go to https://openslide.org/download/#windows-binaries
# 2. Download Windows package
# 3. Change downloaded file name to "openslide-win64"
# 4. Move to "C:/Users/junho" (or some other known path)
# 5. pip install openslide-python
# 6. Run the code below
##################################################
##################################################

slide_path = "./oncofree_sample.tif"
mask_level = 0
OPENSLIDE_PATH = "C:/Users/junho/openslide-win64/bin"

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

try:
    slide = openslide.OpenSlide(slide_path)
except openslide.OpenSlideError:
    print(f"Error: Cannot Open OpenSlide Module. {slide_path}")
    sys.exit()
    
slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[mask_level]))
print(slide_map)