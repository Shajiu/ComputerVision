# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: Rectangle.py
# @Time    : 2021/11/14 15:13

import cv2
import os
import csv
import codecs
import time

def readFile(filePath):
    '''
    :param filePath:        csv文件并存储对应的数据信息
    :return:    字典{"文件名称"：[xmin,ymin,xmax,ymax]}
    '''
    fileDictionaries = dict()
    for index, name in enumerate(os.listdir(filePath)):
        csv_reader = csv.reader(codecs.open(os.path.join(filePath,name)))
        xmin = 1000000000
        ymin = 1000000000
        xmax = 0
        ymax = 0
        coordinate = []
        for line in csv_reader:
            x = int(float(line[0]))
            y = int(float(line[1]))
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
        # 如下添加顺序不能变，切记
        coordinate.append(ymin)
        coordinate.append(xmin)
        coordinate.append(ymax)
        coordinate.append(xmax)

        fileDictionaries[name.split(".")[0]] = coordinate
    return fileDictionaries


def rectangle(fileDictionaries, testPath, rectanglePath):
    '''
    :param fileDictionaries:  字典{"文件名称"：[xmin,ymin,xmax,ymax]}
    :param testPath:          原始测试的图像
    :param rectanglePath:     结果存储的路劲
    :return:
    '''
    for index, name in enumerate(os.listdir(testPath)):
        if name.split(".")[0] in fileDictionaries:
            image = cv2.imread(os.path.join(testPath, name))
            cv2.rectangle(image, (fileDictionaries[name.split(".")[0]][0], fileDictionaries[name.split(".")[0]][1]),
                          (fileDictionaries[name.split(".")[0]][2], fileDictionaries[name.split(".")[0]][3]),
                          (0, 0, 255), 2)
            cv2.imwrite(os.path.join(rectanglePath, name), image)
            # cv2.imshow("Test", image)
            # cv2.waitKey(0)

def main():
    start = time.time()  # 计算开始时间
    # 读取的csv文件路径
    filePath = "./result/csvFile/horse"
    fileDictionaries = readFile(filePath)
    # 读取的测试原始数据集
    testPath = "./ImgData/testData/sourceData"
    # 定位框出之后存储的数据路径
    rectanglePath = "./result/rectangle"
    rectangle(fileDictionaries, testPath, rectanglePath)
    end = time.time()  # 最后的时间
    print(f"画框所用的时间为:{end - start: .2f} seconds","\t数据大小为:",len(fileDictionaries))

if __name__ == '__main__':
    main()