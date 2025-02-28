#####################################################################
######################### INSTALL MODULE ############################
#####################################################################

# 1 PyQt5 Designer: Creates '.ui' file
* Download Anaconda
* Open "Anaconda Prompt"
* Type "Designer" command in prompt


# 2 OpenSlide: Open & control '.tif' file
* Check "openslide_install.py" file for installation guidelines

# 3 vips: Convert slide to pyramidal structure
* VIPS_PATH = "C:/Users/junho/vips/bin"
* os.environ["PATH"] = VIPS_PATH + os.pathsep + POPPLER_PATH + os.pathsep + os.environ["PATH"]
* import vips, subprocess


# 4 PyQt5: 
* "pip install pyqt5"


# 5 etc: "conda list"
* Check "conda_list.txt" file


# 6 PyQt Load: Load PyQt GUI on windows
* "python pathology_viewer_ver3.py" in terminal



### View Ver 1
#### Main Page
 _________________________________________________________________________________
| File                                                                          X |
|---------------------------------------------------------------------------------|
|     [original_label]                             [predicted_label]              |
| ______________________________________     __________________________________   |
| |                                     |    |                                 |  |
| |                                     |    |                                 |  |
| |                                     |    |                                 |  |
| |            <Image Here>             |    |            <Image Here>         |  |
| |                                     |    |                                 |  |
| |                                     |    |                                 |  |
| |                                     |    |                                 |  |
| |                                     |    |                                 |  |
| |                                     |    |                                 |  |
| |                                     |    |                                 |  |
| |                                     |    |                                 |  |
| |                                     |    |                                 |  |
| |                                     |    |                                 |  |
| |                                     |    |                                 |  |
| |_____________________________________|    |_________________________________|  |
|                                                                                 |
|                                                                                 |
|  [Predict Button] [Generate Button]                             [progress bar]  |
|_________________________________________________________________________________|

#### About Page
 _______________________________________________________
| [logo] [Title]                                      X |
|                                                       |
|    ______________________________________________     |
|    |                     |                       |    |
|    |                     |                       |    |
|    |                     |                       |    |
|    |                     |                       |    |
|    |     [logo]          |      [Title]          |    |
|    |                     |                       |    |
|    |                     |                       |    |
|    |                     |                       |    |
|    |_____________________|_______________________|    |
|                  ____________________                 |
|                  | [Author: Tesser]  |                |
|                  | [License: Tesser] |                |
|                  | [Version: Tesser] |                |
|                  |___________________|                |
|    ______________________________________________     |
|    | About                                       |    |
|    |                                             |    |
|    | Pathology AI Viewer is an automated deep    |    |
|    | learning algorithm based tumor mask         |    |
|    | generator. It will automatically predict    |    |
|    | and generate tumor regions for whole slide  |    |
|    | images. Current purpose is for breast       |    |
|    | pathology slides.                           |    |
|    |                                             |    |
|    | Guidelines                                  |    |
|    |                                             |    |
|    | 1. Open image file (jpeg, svs, tif, png)    |    |
|    | from personal directory.                    |    |
|    | 2. Check if image was loaded on to the      |    |
|    | viewer.                                     |    |
|    | 3. Use Predict button to run the DL         |    |
|    | algorithm.                                  |    |
|    | 4. Use Generate button to create solid tumor|    |
|    | mask.                                       |    |
|    |_____________________________________________|    |
|                           [OK]                        |
|_______________________________________________________|