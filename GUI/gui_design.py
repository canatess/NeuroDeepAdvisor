import os

import cv2
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
from PyQt5.QtCore import QCoreApplication

import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


class WindowDesign(QMainWindow):

    def __init__(self):
        super(WindowDesign, self).__init__()
        uic.loadUi("gui_design.ui", self)
        self.setWindowTitle("Alzheimer Detector")
        self.setWindowIcon(QtGui.QIcon('GUI_Images/hacettepe_logo.png'))
        self.setFixedWidth(1009)
        self.setFixedHeight(606)
        self.show()

        # Default image
        self.current_file = "GUI_Images/default.jpg"
        pixmap = QtGui.QPixmap(self.current_file)
        pixmap = pixmap.scaled(441, 441)
        self.mr_image.setPixmap(pixmap)
        self.mr_image.setMinimumSize(1, 1)

        # Options Menu
        self.file_list = None
        self.file_counter = None
        self.actionOpen_Image.triggered.connect(self.open_image)
        self.actionOpen_Directory.triggered.connect(self.open_directory)
        self.actionQuit.triggered.connect(QCoreApplication.instance().quit)
        self.forward_button.clicked.connect(self.next_image)
        self.backward_button.clicked.connect(self.prev_image)

        # Models
        self.output_dict = {0: "MildDemented",
                            1: "ModerateDemented",
                            2: "NonDemented",
                            3: "VeryMildDemented"}
        self.cnn = torch.load("Models/cnn_model_3.pt")
        self.yolo_v5 = YOLO("Models/yolo_v5.pt")
        self.yolo_v8 = YOLO("Models/yolo_v8.pt")

    # Buttons
    def open_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Image Files (*.png, *.jpg", options=options)

        if filename != "":
            self.current_file = filename
            im_pixmap = QtGui.QPixmap(self.current_file)
            im_pixmap = im_pixmap.scaled(441, 441)
            self.mr_image.setPixmap(im_pixmap)

    def open_directory(self):
        directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.file_list = [directory + "/" + f for f in os.listdir(directory) if
                          f.endswith(".jpg") or f.endswith(".png")]
        self.file_counter = 0
        self.current_file = self.file_list[self.file_counter]
        im_pixmap = QtGui.QPixmap(self.current_file)
        im_pixmap = im_pixmap.scaled(441, 441)
        self.mr_image.setPixmap(im_pixmap)
        self.actual(self.current_file)
        self.cnn_testing(self.current_file)
        self.yolo_testing(self.current_file)

    def next_image(self):
        if self.file_counter is not None:
            self.file_counter += 1
            self.file_counter %= len(self.file_list)
            self.current_file = self.file_list[self.file_counter]
            im_pixmap = QtGui.QPixmap(self.current_file)
            im_pixmap = im_pixmap.scaled(441, 441)
            self.mr_image.setPixmap(im_pixmap)
            self.actual(self.current_file)
            self.cnn_testing(self.current_file)
            self.yolo_testing(self.current_file)

    def prev_image(self):
        if self.file_counter is not None:
            self.file_counter -= 1
            self.file_counter %= len(self.file_list)
            self.current_file = self.file_list[self.file_counter]
            im_pixmap = QtGui.QPixmap(self.current_file)
            im_pixmap = im_pixmap.scaled(441, 441)
            self.mr_image.setPixmap(im_pixmap)
            self.actual(self.current_file)
            self.cnn_testing(self.current_file)
            self.yolo_testing(self.current_file)

    # Model Outputs
    def actual(self, image):
        self.actual_result.setText(f"ACTUAL CLASS\n{image.split('_')[0].split('/')[-1]}")
        self.actual_result.setStyleSheet("color: white;""background-color: red")

    def cnn_testing(self, image):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        test_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])
        cnn = self.cnn.eval()
        image = Image.open(image)
        image = test_transformer(image).float().cuda()
        image = image.unsqueeze(0)

        output = cnn(image)

        _, pred = torch.max(output.data, 1)
        self.cnn_result.setText(f"CNN MODEL\n{self.output_dict[pred.item()]}")
        self.cnn_result.setStyleSheet("color: white;""background-color: green")

    def yolo_testing(self, image):

        image = cv2.imread(image)

        v5_results = self.yolo_v5(image)
        v8_results = self.yolo_v8(image)

        v5_results[0].save_txt("v5_result.txt", save_conf=True)
        v8_results[0].save_txt("v8_result.txt", save_conf=True)

        # Read the txt files first character
        with open("v5_result.txt", "r") as f:
            output = f.readline().split(' ')
            self.yolo_result_2.setText(f"YOLO V5 MODEL\n{self.output_dict[int(output[0])]} (%{float(output[5]):.3f})")
            self.yolo_result_2.setStyleSheet("color: white;""background-color: blue")

        with open("v8_result.txt", "r") as f:
            output = f.readline().split(' ')

            self.yolo_result.setText(f"YOLO V8 MODEL\n{self.output_dict[int(output[0])]} (%{float(output[5]):.3f})")
            self.yolo_result.setStyleSheet("color: white;""background-color: blue")

        os.remove('v5_result.txt')
        os.remove('v8_result.txt')


if __name__ == '__main__':
    app = QApplication([])
    app_window = WindowDesign()
    app.exec_()
