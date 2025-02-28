import os, sys, io
import fitz
import threading
import numpy as np
from PIL import Image, ImageQt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtWidgets import QApplication
from PyQt5 import uic
import tifffile as tiff
from PIL import Image
import csv, tempfile, time
from io import BytesIO

# from pipeline import run_pipeline
import pathology_viewer_pipeline_main as pipe

Image.MAX_IMAGE_PIXELS = None
Image.MAX_MEMORY_USAGE = 0

OPENSLIDE_PATH = "C:/Users/junho/openslide-win64/bin"
VIPS_PATH = "C:/Users/junho/vips/bin"
POPPLER_PATH = "C:/Users/junho/poppler-23.07.0/Library/bin"

# install openslide for windows
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

os.environ["PATH"] = VIPS_PATH + os.pathsep + POPPLER_PATH + os.pathsep + os.environ["PATH"]
# import vips
# os.environ["PATH"] = VIPS_PATH + os.pathsep + os.environ["PATH"]
import subprocess

# import poppler
# os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ["PATH"]
import pdf2image
from pdf2image import convert_from_path

# main_class = uic.loadUiType("main_page.ui")[0]
about_class = uic.loadUiType("./design/about_page.ui")[0]

# Custom stream to redirect stdout to both terminal and QTextEdit
class StreamLogger:
    def __init__(self, text_widget, orig_stream):
        self.text_widget = text_widget
        self.orig_stream = orig_stream

    def write(self, message):
        # Write to the original stream (console)
        self.orig_stream.write(message)

        # Write to the QTextEdit widget (terminal_display)
        cursor = self.text_widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(message)
        self.text_widget.setTextCursor(cursor)
        self.text_widget.ensureCursorVisible()

    def flush(self):
        self.orig_stream.flush()


########################################################################################################
########################################################################################################
# load "preview" page
class PreviewDialog(QDialog):
    def __init__(self):
        super().__init__()

    def show_pdf(self, images):
        self.setWindowTitle('PDF Preview')
        app_icon = QIcon("./design/tesser.png")
        self.setWindowIcon(app_icon)
        # self.setGeometry(400, 150, 750, 800)
        self.resize(750, 800)

        # pdf display layout
        layout = QVBoxLayout()
        scroll_area = QScrollArea(self)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setWidgetResizable(True)

        # create a widget to hold the images
        content_widget = QWidget(self)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Use QGraphicsView to display images with resizable feature
        graphics_view = QGraphicsView(self)
        graphics_scene = QGraphicsScene()
        graphics_view.setScene(graphics_scene)
        graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        graphics_view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # Calculate total height of all images for proper positioning
        total_height = 0
        for image in images:
            pixmap = QPixmap(image)
            total_height += pixmap.height()

        current_y = 0
        for image in images:
            pixmap = QPixmap(image)
            item = QGraphicsPixmapItem(pixmap)
            graphics_scene.addItem(item)
            item.setPos(0, current_y)  # Set the position of the item
            current_y += pixmap.height()  # Move the next item below the current one
        
        scroll_area.setWidget(graphics_view)
        layout.addWidget(scroll_area)  # add the scroll area to the main layout

        self.setLayout(layout)

########################################################################################################
########################################################################################################

# load "about" page
class AboutDialog(QDialog, about_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setWindowTitle("About")
        app_icon = QIcon("./design/tesser.png")
        self.setWindowIcon(app_icon)

        # load and display the image & application name
        background_pixmap = QPixmap("./design/microscope.png")
        self.background_image.setPixmap(background_pixmap)
        self.background_image.setAlignment(Qt.AlignCenter)
        opacity_effect = QGraphicsOpacityEffect(self)
        opacity_effect.setOpacity(0.4)
        self.background_image.setGraphicsEffect(opacity_effect)

        self.background_name.setText("Pathology<br>AI<br>Viewer")
        self.background_name.setAlignment(Qt.AlignCenter)
        self.background_name.setStyleSheet('color: black; font-size: 24px; font-weight: bold;')

        self.frame.setLayout(self.background_setter)
        self.frame.setStyleSheet('background-color: rgba(0, 0, 0, 0);')

        # set the text for license, version, and homepage link
        self.license_label.setText("License: Tesser Inc.")
        self.version_label.setText("Version: 1.0 (23.07.20)")
        self.homepage_label.setText("Homepage: https://www.tesser.co.kr/")

        # text for application name and description
        description_text = ("About\n\n"
                            "Pathology AI Viewer is an automated deep learning algorithm based tumor mask generator. "
                            "It will automatically predict and generate tumor regions for whole slide images. "
                            "The current purpose is for breast pathology slides.")
        self.description_text.setPlainText(description_text)
        
        # text for guidelines
        guidelines_text = ("Guidelines\n\n"
                           "Step 1: Open the folder containing the Whole Slide Image (WSI) in the TIFF format.\n"
                           "Step 2: Verify that the image files are loaded by checking the corresponding box.\n"
                           "Step 3: Process the Whole Slide Image using the Deep Learning (DL) model by clicking the 'Predict' button.\n"
                           "Step 4: Generate a PDF version of the processed image by clicking the 'Export' button. The PDF will be saved automatically.")

        self.guidelines_text.setPlainText(guidelines_text)

        # Connect OK button
        self.ok_button.clicked.connect(self.accept)


# load "main" application window
class WindowClass(QMainWindow):
    def __init__(self) :
        super().__init__()
        # self.setupUi(self)

        self.resize(1000, 800)
        self.setWindowTitle("Pathology AI Viewer")
        app_icon = QIcon("./design/tesser.png")
        self.setWindowIcon(app_icon)

        #################################################################################################
        # create menu bar
        menubar = self.menuBar()

        # "File"
        FileMenu = menubar.addMenu("File")
        openfolder = QAction("Open Folder", self)
        closefolder = QAction("Close Folder", self)
        exit = QAction('Exit',self)
        openfolder.triggered.connect(self.openFolder)
        closefolder.triggered.connect(self.closeFolder)
        exit.triggered.connect(qApp.quit)
        FileMenu.addAction(openfolder)
        FileMenu.addAction(closefolder)
        FileMenu.addAction(exit)

        # "Help"
        HelpMenu = menubar.addMenu("Help")
        aboutPage = QAction("About...", self)
        aboutPage.triggered.connect(self.aboutPage)
        HelpMenu.addAction(aboutPage)

        #################################################################################################
        ##### CREATE CENTRAL WIDGET & MAIN LAYOUT
        # central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        # main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)

        ##### LEFT SIDE "UPLOAD" FEATURES
        left_side = QVBoxLayout()
        left_side.setContentsMargins(0, 0, 15, 0)

        # self.tab_widget = QTabWidget(self)
        # self.tab_widget.setFixedHeight(22)
        # font_tab_bar = QFont("Times New Roman", 10)
        # self.tab_widget.setFont(font_tab_bar)
        # self.tab1 = QWidget(self)
        # self.tab2 = QWidget(self)
        # self.tab3 = QWidget(self)
        # self.tab_widget.addTab(self.tab1, "Viewer")
        # self.tab_widget.addTab(self.tab2, "Report")
        # self.tab_widget.addTab(self.tab3, "ETC")
    
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMaximumHeight(90)
        # self.scroll_area.setMinimumHeight(50)
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.file_displayer = QTextEdit(self)
        font = QFont("Times New Roman", 12)
        self.file_displayer.setFont(font)
        self.file_displayer.setStyleSheet("line-height: 120%;")
        self.file_displayer.setReadOnly(True)
        self.file_displayer.setWordWrapMode(QTextOption.NoWrap)
        self.scroll_area.setWidget(self.file_displayer)

        # self.button_predict = QPushButton("Run")
        # font_click = QFont("Times New Roman", 12)
        # self.button_predict.setFont(font_click)
        # self.button_predict.setStyleSheet("border: 3px solid black;")
        # self.button_predict.setMinimumSize(120, 45)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(40)

        self.log_area = QScrollArea(self)
        self.log_area.setWidgetResizable(True)
        self.log_area.setMaximumHeight(140)
        # self.log_area.setMinimumHeight(130)
        self.terminal_display = QTextEdit(self)
        font = QFont("Fira Code", 10)
        self.terminal_display.setFont(font)
        self.terminal_display.setStyleSheet("line-height: 120%;")
        self.terminal_display.setReadOnly(True)
        self.terminal_display.setWordWrapMode(QTextOption.NoWrap)
        self.log_area.setWidget(self.terminal_display)

        # Redirect stdout to update the QTextEdit
        sys.stdout = StreamLogger(self.terminal_display, sys.stdout)




        right_side_button_layout = QHBoxLayout()
        # right_side_button_layout = QVBoxLayout()

        click_buttons_layout = QVBoxLayout()
        radio_button1 = QRadioButton("Soft")
        radio_button2 = QRadioButton("Moderate")
        radio_button3 = QRadioButton("Hard")
        radio_button1.setChecked(True) # Set radio_button1 as the default selected radio button
        font = QFont("Times New Roman", 12, QFont.Bold)
        font.setWeight(QFont.Bold)
        radio_button1.setFont(font)
        radio_button2.setFont(font)
        radio_button3.setFont(font)

        click_buttons_layout.addWidget(radio_button1)
        click_buttons_layout.addWidget(radio_button2)
        click_buttons_layout.addWidget(radio_button3)

        radio_button1.clicked.connect(self.onRadioButtonClicked)
        radio_button2.clicked.connect(self.onRadioButtonClicked)
        radio_button3.clicked.connect(self.onRadioButtonClicked)

        self.track_option = "Soft"

        # show buttons
        show_buttons_layout = QVBoxLayout()
        self.button_predict = QPushButton("Run")
        font_click = QFont("Times New Roman", 12)
        self.button_predict.setFont(font_click)
        self.button_predict.setStyleSheet("border: 3px solid black;")
        self.button_predict.setMinimumSize(120, 45)
        self.button_preview = QPushButton("Preview PDF")
        self.button_export = QPushButton("Export PDF")
        # font
        self.button_preview.setFont(font_click)
        self.button_export.setFont(font_click)
        # style
        self.button_preview.setStyleSheet("border: 3px solid black;")
        self.button_export.setStyleSheet("border: 3px solid black;")
        # size
        self.button_preview.setMinimumSize(120, 45)
        show_buttons_layout.addWidget(self.button_predict)
        show_buttons_layout.addSpacing(15)
        self.button_export.setMinimumSize(120, 45)
        show_buttons_layout.addWidget(self.button_preview)
        show_buttons_layout.addSpacing(15)
        show_buttons_layout.addWidget(self.button_export)

        # right_side_button_layout.addSpacing(45)
        right_side_button_layout.addLayout(show_buttons_layout)
        right_side_button_layout.addLayout(click_buttons_layout)
        right_side_button_layout.addStretch()


        # right_side_button_layout.addWidget(radio_button1)
        # right_side_button_layout.addSpacing(5)
        # right_side_button_layout.addWidget(radio_button2)
        # right_side_button_layout.addSpacing(10)
        # right_side_button_layout.addWidget(radio_button3)
        # right_side_button_layout.addSpacing(5)

        self.label_file = QLabel("File Names")
        self.label_log = QLabel("Terminal")
        self.label_options = QLabel("User Options")
        font_label = QFont("Times New Roman", 14, QFont.Bold)
        font_label.setWeight(QFont.Bold)
        self.label_file.setFont(font_label)
        self.label_log.setFont(font_label)
        self.label_options.setFont(font_label)


        # left_side.addSpacing(45)
        # left_side.addWidget(self.tab_widget)
        left_side.addSpacing(13)
        left_side.addWidget(self.label_file)
        left_side.addSpacing(5)
        left_side.addWidget(self.scroll_area)
        left_side.addSpacing(10)
        left_side.addWidget(self.label_log)
        left_side.addSpacing(5)
        left_side.addWidget(self.log_area)
        left_side.addSpacing(10)
        left_side.addWidget(self.label_options)
        left_side.addSpacing(5)
        left_side.addLayout(right_side_button_layout)
        # left_side.addWidget(self.button_predict)
        # left_side.addSpacing(10)
        # left_side.addWidget(self.progress_bar)
        # left_side.addSpacing(20)
        # left_side.addWidget(self.log_area)
        left_side.addStretch()

        ##### RIGHT SIDE "SHOW" FEATURES
        right_side = QVBoxLayout()
        right_side.setContentsMargins(15, 0, 0, 0)

        ##### RIGTH SIDE TOP SECTION DROPDOWN
        # addWidget "drop down selection bar"
        self.combo_box = QComboBox()
        self.combo_box.setFixedHeight(25)
        self.combo_box.addItem("--select option--")
        self.combo_box.currentIndexChanged.connect(self.onComboBoxIndexChanged)

        ##### RIGTH SIDE MIDDLE SECTION VIEWER
        # addWidget "show resized whole slide image (cropped)"
        self.predicted_image_viewer = QGraphicsView(self)

       
        ##### COMBINE TOP/MIDDLE/BOTTON SECTIONS
        right_side.addWidget(self.combo_box)
        right_side.addSpacing(15)
        right_side.addWidget(self.predicted_image_viewer)
        right_side.addSpacing(15)
        right_side.addWidget(self.progress_bar)
        # right_side.addSpacing(15)
        # right_side.addLayout(right_side_button_layout)

        ##### COMBINE LEFT AND RIGHT SIDE
        main_layout.addLayout(left_side, 2)
        main_layout.addLayout(right_side, 5)

        # set up QGraphicsScenes for the QGraphicsViews
        self.scene_predicted = QGraphicsScene()
        self.predicted_image_viewer.setScene(self.scene_predicted)

        # If needed, enable scroll bars in the viewer
        self.predicted_image_viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.predicted_image_viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # predict & generate buttons
        self.button_predict.clicked.connect(self.buttonPredictionFunction) # runs DL model
        self.button_preview.clicked.connect(self.buttonPreviewFunction) # shows preview
        self.button_export.clicked.connect(self.buttonExportFunction) # export as PDF
        # self.progress_bar.valueChanged.connect(self.printValue) # progress bar
        self.progress_bar.setValue(0) # initializes progress bar to 0


        # dictionary to store the output images from pipeline.py
        self.folder_path = ""
        self.output_images = {}
        self.pdf_report = None
        self.selected_option = "Soft"
        #################################################################################################
        #################################################################################################
    
    """
    open folder and show file names
    """
    def openFolder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            file_names = [file for file in os.listdir(folder_path) if file.endswith('.tiff') or file.endswith('.tif')]
            if len(file_names) > 30:
                QMessageBox.warning(self, "Max Limit Exceeded", "The maximum number of files allowed in the folder is 30.")
            else:
                self.folder_path = folder_path
                self.combo_box.clear()
                self.showFileNames(folder_path)


    def showFileNames(self, folder_path):
        file_names = [file for file in os.listdir(folder_path) if file.endswith('.tiff') or file.endswith('.tif')]
        self.file_displayer.setPlainText('\n'.join(file_names))
        self.combo_box.clear()
        self.combo_box.addItems(file_names)


    def ndarray_to_qimage(self, arr):
        height, width, channel = arr.shape
        bytes_per_line = 3 * width
        qimage = QImage(arr.data, width, height, bytes_per_line, QImage.Format_RGB888)

        return qimage.copy()


    def onComboBoxIndexChanged(self):
        try: 
            selected_file = self.combo_box.currentText()
            if selected_file in self.output_images and self.selected_option in self.output_images[selected_file]:
                pixmap = QPixmap.fromImage(self.ndarray_to_qimage(self.output_images[selected_file][self.selected_option]))
                self.scene_predicted.clear()  # Clear the existing scene before adding a new pixmap item
                pixmap_item = QGraphicsPixmapItem(pixmap)
                self.scene_predicted.addItem(pixmap_item)

                self.scene_predicted.setSceneRect(pixmap_item.boundingRect())  # Adjust the scene size to fit the pixmap
                self.predicted_image_viewer.fitInView(self.scene_predicted.sceneRect(), Qt.KeepAspectRatio)  # Re-scale the view to fit the pixmap
                self.predicted_image_viewer.setScene(self.scene_predicted)  # Update the view
        except TypeError as e:
            print("Error:", e)


    def onRadioButtonClicked(self):
        try:
            self.selected_option = self.sender().text()
            selected_file = self.combo_box.currentText()
            if selected_file in self.output_images and self.selected_option in self.output_images[selected_file]:
                pixmap = QPixmap.fromImage(self.ndarray_to_qimage(self.output_images[selected_file][self.selected_option]))
                self.scene_predicted.clear()
                pixmap_item = QGraphicsPixmapItem(pixmap)
                self.scene_predicted.addItem(pixmap_item)

                self.scene_predicted.setSceneRect(pixmap_item.boundingRect())  
                self.predicted_image_viewer.fitInView(self.scene_predicted.sceneRect(), Qt.KeepAspectRatio)  
                self.predicted_image_viewer.setScene(self.scene_predicted) 
        except TypeError as e:
            print("Error:", e)
            

    def closeFolder(self):
        self.scene_predicted.clear()
        self.combo_box.clear()
        self.combo_box.addItem("--select option--")
        self.combo_box.setCurrentIndex(0)
        self.file_displayer.clear()
        self.terminal_display.clear()
        self.predicted_image_viewer.scene().clear()
        self.progress_bar.setValue(0)


    def run_pipe_main(self):
        self.output_images, self.pdf_report = pipe.main(self.folder_path, self.updateProgressBar)

    def buttonPredictionFunction(self):
        print("## Start Prediction")
        prediction_thread = threading.Thread(target=self.run_pipe_main)
        prediction_thread.daemon = True

        prediction_thread.start()
        self.onComboBoxIndexChanged()
        print("## Select Your Option")

    def updateProgressBar(self, progress):
        self.progress_bar.setValue(progress)
        QApplication.processEvents()

    """
    load "about" page
    """
    def aboutPage(self):
        about_dialog = AboutDialog()
        about_dialog.exec_()

    """
    error handling
    """
    def showErrorMessage(self, title, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()


###########################################################################################################
###########################################################################################################
    def buttonPreviewFunction(self):
        print("Preview Button Clicked")

        pdf = self.pdf_report
        images = self.capture_images_from_pdf(pdf)
        if not images:
            print("Error: No images captured from PDF.")
            return
        
        preview_window = PreviewDialog()
        preview_window.show_pdf(images)
        preview_window.exec_()


    def capture_images_from_pdf(self, file_path):
        images = []

        try:
            pdf_file_like_object = io.BytesIO(self.pdf_report)
            pdf_document = fitz.open(stream=pdf_file_like_object)

            for page_number in range(pdf_document.page_count):
                page = pdf_document.load_page(page_number)
                # render the page as a QImage
                img = page.get_pixmap()
                qimg = QImage(img.samples, img.width, img.height, img.stride, QImage.Format_RGB888)
                # save the QImage as a PNG
                temp_image_path = f"page_{page_number}.png"
                qimg.save(temp_image_path, "PNG")
                images.append(temp_image_path)

        except Exception as e:
            print("Error:", e)

        return images


    def buttonExportFunction(self):
        export_report = self.pdf_report
        if not export_report:
            print("404 File Not Found")
            return
        
        print("Export Button Clicked")

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "", "PDF Files (*.pdf);;All Files (*)", options=options)
        if not file_path: # User canceled saving
            return
        
        try:
            with open(file_path, "wb") as file:
                file.write(export_report)
            print("PDF Exported Successfully.")
        except Exception as e:
            print("Error occurred while exporting PDF:", str(e))


if __name__ == "__main__" :
    app = QApplication(sys.argv)  # runs QApplication
    myWindow = WindowClass() # creates WindowClass instance
    myWindow.show() # shows program window
    app.exec_() 
