# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: trainUnit.py
# @Time    : 2021/11/7 11:42
from unit import ObjectDetection
'''
功能:   训练部分的主程序
'''
def train():
    '''
    类实例化
    :return:
    '''
    bart = ObjectDetection()
    # 原始数据文件夹
    dataPath = "./ImgData/referenceData/sourceData"
    # 调用遍历文件函数
    sourcePathDictionaries = bart.get_files(dataPath)
    # 存储分割并二值化后的文件路径
    dataBinaryzation = "./ImgData/referenceData/BinaryDataSets"
    # 分割模型路径
    semanticModel = "./models/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
    # 调用分割与二值化函数
    bart.semanticBinaryzation(semanticModel, dataBinaryzation, sourcePathDictionaries)
    # 提取感兴趣的区域图路径
    interestedData="./ImgData/referenceData/InterestedData"
    # 调用遍历文件并获取分割二值化后的文件
    dataBinaryzationDictionaries=bart.get_files(dataBinaryzation)
    # 调用提取感兴趣区域的函数
    bart.extraction(sourcePathDictionaries,dataBinaryzationDictionaries,interestedData)
    # 存储边缘图的文件路径
    cannyPath="./ImgData/referenceData/cannyDatas"
    # 调用遍历文件并获取感兴趣区域后的文件
    interestedDataDictionaries=bart.get_files(interestedData)
    # 调用针对感兴趣区域进行边缘提取的函数
    bart.canny(interestedDataDictionaries, cannyPath)

if __name__ == '__main__':
    train()
