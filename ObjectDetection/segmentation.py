# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: segmentation.py
# @Time    : 2021/10/4 09:26

from pixellib.instance import instance_segmentation
import time
import pandas as pd
import os

segment_image = instance_segmentation()
def read_files(filePath):
    names=[] # 文件名称
    timeList=[]  # 花费时间
    start = time.time()  # 起始时间
    for name in os.listdir(filePath):
        segment_image.load_model("./models/mask_rcnn_coco.h5")
        names.append(name)
        t=segmentImage(name,filePath)
        timeList.append(t)
    end = time.time()  # 终止时间
    print(f"pixellib检测所用的时间:{end - start: .2f} seconds", "\t数据大小为:", len(names))
    dataframe = pd.DataFrame({'文件名称': names,"分割时间":timeList})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("./result/segmentImageTime/segmentime.csv", index=False, sep=',',encoding="utf-8")

def segmentImage(name,filePath):
    s = time.time()
    segment_image.segmentImage(os.path.join(filePath,name), output_image_name=os.path.join("./result/segmentImage/",name),show_bboxes = True)
    e=time.time()
    return f"{e - s: .2f} seconds"

if __name__ == '__main__':
    read_files(filePath="./ImgData/testData/sourceData")
