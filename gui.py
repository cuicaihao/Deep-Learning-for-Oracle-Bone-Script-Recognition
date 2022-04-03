# Conver the **.ui to **.py file and then run this file to test the GUI
# Example:
# pyside6-uic obs_gui_v1.ui -o obs_gui.py

import sys
# from PySide6.QtWidgets import (QApplication, QWidget, QFrame, QFileDialog)
# from PySide6.QtGui import (QPainter, QPen)
# from PySide6.QtCore import Qt

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
import sys
import numpy as np
from torch import true_divide

from src.ui.obs_gui import *
from src.models.predict_model import *

# from translate import Translator


class ScribbleArea(QFrame):  #

    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)

        #resize设置宽高，move设置位置
        self.resize(250, 257)
        # self.move(100, 100)
        # self.setWindowTitle("简单的画板4.0")

        #setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)
        '''
            要想将按住鼠标后移动的轨迹保留在窗体上
            需要一个列表来保存所有移动过的点
        '''

        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        # variables
        # drawing flag
        self.drawing = False
        # default brush size
        self.brushSize = 6
        # default color
        self.brushColor = Qt.black

        # QPoint object to tract the point
        self.lastPoint = QPoint()

    # method for checking mouse cicks
    def mousePressEvent(self, event):

        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            # make drawing flag true
            self.drawing = True
            # make last point to the point of cursor
            self.lastPoint = event.position()

    # method for tracking mouse activity
    def mouseMoveEvent(self, event):

        # checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:

            # creating painter object
            painter = QPainter(self.image)

            # set the pen of the painter
            painter.setPen(
                QPen(self.brushColor, self.brushSize, Qt.SolidLine,
                     Qt.RoundCap, Qt.RoundJoin))

            # draw line from the last point of cursor to the current point
            # this will draw only one step
            painter.drawLine(self.lastPoint, event.position())

            # change the last point
            self.lastPoint = event.position()
            # update
            self.update()

    # method for mouse left button release
    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.drawing = False

    # paint event
    def paintEvent(self, event):
        # create a canvas
        canvasPainter = QPainter(self)

        # draw rectangle on the canvas
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def clearImage(self):
        self.image.fill(qRgb(255, 255, 255))
        self.modified = True
        self.update()

    # method for saving canvas
    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        if filePath == "":
            return
        self.image.save(filePath)

    def QImageToCvMat(self):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = self.image.convertToFormat(QImage.Format.Format_RGB32)

        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.constBits()
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        return arr


class MyForm(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(MyForm, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.scribbleArea = ScribbleArea()
        self.scribbleArea.setParent(self.ui.frame_scribble)

        # Run Button Clicked
        self.ui.pushButton_run.clicked.connect(self.run)
        # Clear Button Clicked
        self.ui.pushButton_clean.clicked.connect(self.clean)
        # Translate Button Clicked
        self.ui.pushButton_translate.clicked.connect(self.translate)

        # init the Text of the label
        self.ui.label_prediction.setText("Prediction ID:")
        self.ui.label_english.setText("English:")
        self.ui.label_chinese.setText("Chinese 中文: ")

        # self.translator = Translator(from_lang="zh-cn", to_lang="english")

        model_path = './models/model_best.pt'
        label_path = './data/processed/label_name.csv'
        # load the model
        self.model = PredictModel(model_path, label_path)

        self.x = 0
        self.show()

    def translate(self):
        # TO-DO Chinese to English Translation
        return

    def clean(self):
        self.ui.label_prediction.setText("Prediction ID:")
        self.ui.label_english.setText("English:")
        self.ui.label_chinese.setText("Chinese 中文: ")
        self.scribbleArea.clearImage()
        self.ui.label_chinese.setStyleSheet(
            "background-color: lightgrey; border: 1px solid black;")
        self.ui.lineEdit_chinese.setText("")

    def ai_predict(self, image_path):

        prediction_top_10 = self.model.predict_top10(image_path)
        print(prediction_top_10)
        return prediction_top_10

    def run(self):
        self.x = self.x + 1
        print("Run Button Clicked {}".format(self.x))

        # self.ui.frame_scribble.update()
        # image = self.scribbleArea.image
        # print(type(image))
        image_np = self.scribbleArea.QImageToCvMat()
        # print(type(image_np), image_np.shape)

        # save the image
        # self.scribbleArea.save()
        # with open('test.npy', 'wb') as f:
        #     np.save(f, image_np)
        # import matplotlib.pylab as plt
        # plt.imshow(image_np)
        # plt.show()

        prediction_top_10 = self.ai_predict(image_np)
        top_10_character = prediction_top_10.name.values

        self.ui.lineEdit_chinese.setText(' '.join(top_10_character))

        pred_label = prediction_top_10.iloc[0, 0]
        pred_character = prediction_top_10.iloc[0, 1]
        pred_prob = prediction_top_10.iloc[0, 3]

        self.ui.label_prediction.setText(
            f"Prediction ID: {pred_label} \nAcc: {pred_prob:.8f}")
        self.ui.label_chinese.setText("Chinese 中文: " + pred_character)
        if pred_prob > 0.5:
            self.ui.label_chinese.setStyleSheet(
                "background-color: lightgreen; border: 1px solid black;")

        elif pred_prob < 0.5 and pred_prob > 0.0001:
            self.ui.label_chinese.setStyleSheet(
                "background-color: lightyellow; border: 1px solid black;")
        else:
            self.ui.label_chinese.setStyleSheet(
                "background-color: red; border: 1px solid black;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec())
