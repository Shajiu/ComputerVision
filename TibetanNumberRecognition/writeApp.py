# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: writeApp.py
# @Time    : 2022/1/28 09:03
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QFont
from PIL import ImageGrab, Image
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QLabel
import sys
from PyQt5.QtWidgets import QApplication
import cv2
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import os


class MyMnistWindow(QWidget):

    def __init__(self):
        super(MyMnistWindow, self).__init__()

        self.resize(285, 330)  # 设置窗口宽高
        self.move(100, 100)  # 设置窗口出现时所处于屏幕的位置
        self.setWindowFlags(Qt.FramelessWindowHint)  # 窗体无边框
        # setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)

        self.pos_xy = []  # 保存鼠标移动过的点

        # 添加一系列控件
        # 创建标签控件对象，这里是空的
        self.label_draw = QLabel('', self)
        # 从屏幕上（2，2）位置开始，显示一个280 * 280 的界面（宽高）
        self.label_draw.setGeometry(2, 2, 280, 280)
        # 设置样式
        self.label_draw.setStyleSheet("QLabel{border:2px solid black;}")
        # 消除空隙
        self.label_draw.setAlignment(Qt.AlignCenter)

        self.label_result_name = QLabel('结果：', self)
        self.label_result_name.setGeometry(2, 290, 60, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(64, 290, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:2px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_recognize = QPushButton("识别", self)
        self.btn_recognize.setGeometry(110, 290, 50, 35)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)

        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.setGeometry(170, 290, 50, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        self.btn_close = QPushButton("关闭", self)
        self.btn_close.setGeometry(230, 290, 50, 35)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 5, Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                # 判断是否是断点
                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        '''
            按住鼠标移动事件：将当前点添加到pos_xy列表中
        '''
        # 中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        # pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
            鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

    # 识别按钮的功能：截屏手写数字并将截图转换成28*28像素的图片，之后调用识别函数并显示识别结果
    def btn_recognize_on_clicked(self):
        bbox = (104, 104, 380, 380)
        im = ImageGrab.grab(bbox)  # 截屏，手写数字部分
        im = im.resize((32, 32), Image.ANTIALIAS)  # 将截图转换成 28 * 28 像素
        path = "./data/TEST/test.png"
        im.save(path)
        self.label_result.setText(self.recognize_img(path))  # 显示识别结果
        self.update()

    # 清除按钮的功能：列表置空，识别结果一栏清空
    def btn_clear_on_clicked(self):
        self.pos_xy = []
        self.label_result.setText('')
        self.update()

    # 关闭按钮的功能：关闭窗口
    def btn_close_on_clicked(self):
        self.close()

    # 识别函数
    def recognize_img(self, path):  # 手写体识别函数
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load('./models/model-mnist.pth')  # 加载模型
        model = model.to(device)
        model.eval()  # 把模型转为test模式

        # 读取要预测的图片
        img = cv2.imread(path)
        os.remove(path)
        img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)
        plt.imshow(img, cmap="gray")  # 显示图片
        plt.axis('off')  # 不显示坐标轴

        # 导入图片，图片扩展后为[1，1，32，32]
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为mnist数据集都是灰度图
        img = trans(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]

        # 预测
        output = model(img)
        predict = output.argmax(dim=1)
        return str(predict.item())

       


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mymnist = MyMnistWindow()
    mymnist.show()
    app.exec_()
