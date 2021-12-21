# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: calculate.py
# @Time    : 2021/12/1 21:37
import os

file_name = "E:\Python_Projects\ObjectDetection\ImgData\\testData\sourceData\\000175.jpg"

file_stats = os.stat(file_name)


print(file_stats)
print(f'File Size in Bytes is {file_stats.st_size}')
print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')