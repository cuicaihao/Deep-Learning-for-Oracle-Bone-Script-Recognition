# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'obs_gui.ui'
##
## Created by: Qt User Interface Compiler version 6.2.4
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
                            QMetaObject, QObject, QPoint, QRect, QSize, QTime,
                            QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
                           QFontDatabase, QGradient, QIcon, QImage,
                           QKeySequence, QLinearGradient, QPainter, QPalette,
                           QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMenuBar,
                               QPushButton, QSizePolicy, QStatusBar,
                               QVBoxLayout, QWidget)


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(580, 350)
        MainWindow.setMinimumSize(QSize(580, 350))
        MainWindow.setMaximumSize(QSize(580, 350))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox_draw = QGroupBox(self.centralwidget)
        self.groupBox_draw.setObjectName(u"groupBox_draw")
        self.frame_scribble = QFrame(self.groupBox_draw)
        self.frame_scribble.setObjectName(u"frame_scribble")
        self.frame_scribble.setEnabled(True)
        self.frame_scribble.setGeometry(QRect(10, 30, 250, 250))
        self.frame_scribble.setMinimumSize(QSize(250, 250))
        self.frame_scribble.setMaximumSize(QSize(250, 250))
        self.frame_scribble.setBaseSize(QSize(200, 200))
        self.frame_scribble.setFrameShape(QFrame.StyledPanel)
        self.frame_scribble.setFrameShadow(QFrame.Raised)

        self.horizontalLayout.addWidget(self.groupBox_draw)

        self.groupBox_ctrl = QGroupBox(self.centralwidget)
        self.groupBox_ctrl.setObjectName(u"groupBox_ctrl")
        self.groupBox_ctrl.setMinimumSize(QSize(270, 288))
        self.groupBox_ctrl.setMaximumSize(QSize(270, 288))
        self.verticalLayout = QVBoxLayout(self.groupBox_ctrl)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_prediction = QLabel(self.groupBox_ctrl)
        self.label_prediction.setObjectName(u"label_prediction")

        self.verticalLayout.addWidget(self.label_prediction)

        self.label_chinese = QLabel(self.groupBox_ctrl)
        self.label_chinese.setObjectName(u"label_chinese")

        self.verticalLayout.addWidget(self.label_chinese)

        self.lineEdit_chinese = QLineEdit(self.groupBox_ctrl)
        self.lineEdit_chinese.setObjectName(u"lineEdit_chinese")

        self.verticalLayout.addWidget(self.lineEdit_chinese)

        self.pushButton_clean = QPushButton(self.groupBox_ctrl)
        self.pushButton_clean.setObjectName(u"pushButton_clean")

        self.verticalLayout.addWidget(self.pushButton_clean)

        self.pushButton_run = QPushButton(self.groupBox_ctrl)
        self.pushButton_run.setObjectName(u"pushButton_run")

        self.verticalLayout.addWidget(self.pushButton_run)

        self.label_english = QLabel(self.groupBox_ctrl)
        self.label_english.setObjectName(u"label_english")

        self.verticalLayout.addWidget(self.label_english)

        self.pushButton_translate = QPushButton(self.groupBox_ctrl)
        self.pushButton_translate.setObjectName(u"pushButton_translate")

        self.verticalLayout.addWidget(self.pushButton_translate)

        self.horizontalLayout.addWidget(self.groupBox_ctrl)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 580, 24))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate(
                "MainWindow", u"Oracle Bone Script Recognition (v0.9)", None))
        self.groupBox_draw.setTitle(
            QCoreApplication.translate("MainWindow", u"Input Drawing", None))
        self.groupBox_ctrl.setTitle(
            QCoreApplication.translate("MainWindow", u"Control Panel", None))
        self.label_prediction.setText(
            QCoreApplication.translate("MainWindow", u"Prediction", None))
        self.label_chinese.setText(
            QCoreApplication.translate("MainWindow", u"Chinese", None))
        self.lineEdit_chinese.setText(
            QCoreApplication.translate("MainWindow", u"Other Candidates",
                                       None))
        self.pushButton_clean.setText(
            QCoreApplication.translate("MainWindow", u"Clean", None))
        self.pushButton_run.setText(
            QCoreApplication.translate("MainWindow", u"Run", None))
        self.label_english.setText(
            QCoreApplication.translate("MainWindow", u"English", None))
        self.pushButton_translate.setText(
            QCoreApplication.translate("MainWindow", u"Translate", None))

    # retranslateUi
