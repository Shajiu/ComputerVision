# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: image_jpg_png.py
# @Time    : 2022/1/24 19:59
import PIL.Image
import os
from tqdm import tqdm
from time import sleep
'''
功能：将jpg文件转换为png文件
'''
def jpg_png(path, savepath, name):
    im = PIL.Image.open(path)
    if not os.path.exists(savepath.replace("jpg", "png")):
        os.makedirs(savepath.replace("jpg", "png"))
    im.save(os.path.join(savepath.replace("jpg", "png") ,name.split(".")[0] + ".png"))
    sleep(0.05)

if __name__ == '__main__':
    g = os.walk(r"F:\网上资料\数据文件\藏文手写数字_压缩版\jpg\\")
    for path, dir_list, file_list in tqdm(g):
        sleep(0.25)
        for file_name in tqdm(file_list):
            list = [x.strip() for x in path.split("\\") if x.strip() != '']
            jpg_png(os.path.join(path, file_name), os.path.join('\\'.join(list)), file_name)
            sleep(0.05)
