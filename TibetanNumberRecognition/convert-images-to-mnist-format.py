# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: convert-images-to-mnist-format.py
# @Time    : 2022/1/24 16:09
import os
from PIL import Image
from array import *
from random import shuffle

'''
功能：将png格式的数据转换为二进制个数据
'''
# Load from and save to
Names = [['F:\网上资料\数据文件\藏文手写数字_压缩版\png', 'train'], ["F:\网上资料\数据文件\藏文手写数字_压缩版\png", "test"]]

for name in Names:
    data_image = array('B')
    data_label = array('B')
    FileList = []
    for dirname in os.listdir(name[0])[1:]:  # [1:] Excludes .DS_Store from Mac OS
        path = os.path.join(name[0], dirname)
        for filename in os.listdir(path):
            if filename.endswith(".png"):
                FileList.append(os.path.join(name[0], dirname, filename))

    shuffle(FileList)  # Usefull for further segmenting the validation set

    for filename in FileList:
        label = int(filename.split('\\')[-2])
        Im = Image.open(open(filename, 'rb'))
        width, height = Im.size
        pixel = Im.load()
        for x in range(0, width):
            for y in range(0, height):
                print("像素值:", pixel[y, x])
                data_image.append(pixel[y, x][0])
        data_label.append(label)  # labels start (one unsigned byte each)

    hexval = "{0:#0{1}x}".format(len(FileList), 6)  # number of files in HEX

    header = array('B')
    header.extend([0, 0, 8, 1, 0, 0])
    header.append(int('0x' + hexval[2:][:2], 16))
    header.append(int('0x' + hexval[2:][2:], 16))

    data_label = header + data_label

    if max([width, height]) <= 256:
        header.extend([0, 0, 0, width, 0, 0, 0, height])
    else:
        raise ValueError('Image exceeds maximum size: 256x256 pixels')

    header[3] = 3  # Changing MSB for image data (0x00000803)

    data_image = header + data_image

    output_file = open("./corpus//" + 't10k-images-idx3-ubyte', 'wb')
    data_image.tofile(output_file)
    output_file.close()

    output_file = open("./corpus//" + 't10k-labels-idx1-ubyte', 'wb')
    data_label.tofile(output_file)
    output_file.close()

# gzip resulting files

for name in Names:
    os.system('gzip ' + "./corpus//" + 't10k-images-idx3-ubyte')
    os.system('gzip ' + "./corpus//" + 't10k-labels-idx1-ubyte')
