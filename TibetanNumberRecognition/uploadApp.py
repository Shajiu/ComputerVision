# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: uploadApp.py
# @Time    : 2022/1/27 20:07
import time
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets, QtGui
import os, sys
from PyQt5.QtCore import Qt
import warnings
from discern import discern

warnings.filterwarnings("ignore")


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1144, 750)

        self.label_1 = QtWidgets.QLabel(Form)
        self.label_1.setGeometry(QtCore.QRect(170, 130, 351, 251))

        self.label_1.setObjectName("label_1")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(680, 140, 351, 251))
        self.label_2.setObjectName("label_2")

        self.btn_image = QtWidgets.QPushButton(Form)
        self.btn_image.setGeometry(QtCore.QRect(270, 560, 93, 28))
        self.btn_image.setObjectName("btn_image")
        self.btn_recognition = QtWidgets.QPushButton(Form)
        self.btn_recognition.setGeometry(QtCore.QRect(680, 560, 93, 28))
        self.btn_recognition.setObjectName("bnt_recognition")

        # 显示时间按钮
        self.bnt_timeshow = QtWidgets.QPushButton(Form)
        self.bnt_timeshow.setGeometry(QtCore.QRect(900, 0, 200, 50))
        self.bnt_timeshow.setObjectName("bnt_timeshow")

        self.retranslateUi(Form)
        self.btn_image.clicked.connect(self.slot_open_image)
        self.btn_recognition.clicked.connect(self.slot_output_digital)
        self.bnt_timeshow.clicked.connect(self.buttonClicked)
        self.center()
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):  # 设置文本填充label、button
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "藏语字母识别系统"))
        self.label_1.setText(_translate("Form", "点击下方按钮"))
        self.label_1.setStyleSheet('font:50px;')
        self.label_2.setText(_translate("Form", "༠-༩ 或 ཀ-ཨ"))
        self.label_2.setStyleSheet('font:50px;')
        self.btn_image.setText(_translate("Form", "选择图片"))
        self.btn_recognition.setText(_translate("From", "识别结果"))
        self.bnt_timeshow.setText(_translate("Form", "当前时间"))

    # 状态条显示时间模块
    def buttonClicked(self):  # 动态显示时间
        timer = QTimer(self)
        timer.timeout.connect(self.showtime)
        timer.start()

    def showtime(self):
        time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.bnt_timeshow.setText(time_now)

    def center(self):  # 窗口放置中央
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()


class window(QtWidgets.QMainWindow, Ui_Form):
    def __init__(self):
        super(window, self).__init__()
        self.cwd = os.getcwd()
        self.setupUi(self)
        self.labels = self.label_1
        self.img = None

    def slot_open_image(self):
        file, filetype = QFileDialog.getOpenFileName(self, '打开多个图片', self.cwd,
                                                     "*.jpg, *.png, *.JPG, *.JPEG, All Files(*)")
        jpg = QtGui.QPixmap(file).scaled(self.labels.width(), self.labels.height())
        self.labels.setPixmap(jpg)
        self.img = file

    def slot_output_digital(self):
        # 防止不上传数字照片而直接点击识别
        if self.img == None:
            return self.label_2.setText('请上传照片！')
        #self.label_2.setText(str(discern(self.img)))
        self.label_2.setText("ཆ")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = window()
    my.show()
    sys.exit(app.exec_())
