# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: testUnit.py
# @Time    : 2021/11/7 11:41
# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: trainUnit.py
# @Time    : 2021/11/7 11:42
from unit import ObjectDetection
import time
'''
功能:   训练部分的主程序
'''
def test():
    '''
    类实例化
    :return:
    '''
    bart = ObjectDetection()
    # 原始数据文件夹
    dataPath = "./ImgData/testData/sourceData"
    # 调用遍历文件函数
    sourcePathDictionaries = bart.get_files(dataPath)
    # 存储分割并二值化后的文件路径
    dataBinaryzation = "./ImgData/testData/BinaryDataSets"
    # 分割模型路径
    semanticModel = "./models/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
    # 调用分割与二值化函数
    bart.semanticBinaryzation(semanticModel, dataBinaryzation, sourcePathDictionaries)
    # 第一次过滤，匹配图像分割并二值化后的Hu值
    # 设置的阈值
    threshold=0.08
    # 调用遍历文件并获取分割二值化后的文件
    dataBinaryzationDictionaries = bart.get_files(dataBinaryzation)

    resultHu1="./ImgData/testData/resultHu1"
    referencePath=bart.get_files("./ImgData/referenceData/BinaryDataSets")
    bart.calculateHu(referencePath, dataBinaryzationDictionaries, threshold,resultHu1)

    # 提取感兴趣的区域图路径
    interestedData = "./ImgData/testData/InterestedData"
    # 调用提取感兴趣区域的函数
    resultHu1Dictionaries = bart.get_files(resultHu1)
    bart.extraction(sourcePathDictionaries,resultHu1Dictionaries,interestedData)
    # 存储边缘图的文件路径
    cannyPath="./ImgData/testData/cannyDatas"
    # 调用遍历文件并获取感兴趣区域后的文件
    interestedDataDictionaries = bart.get_files(interestedData)
    # 调用针对感兴趣区域进行边缘提取的函数
    bart.canny(interestedDataDictionaries, cannyPath)

    # 第二次匹配过滤
    threshold = 0.6
    resultHu2 = "./ImgData/testData/resultHu2"
    refcannyDatasDataDictionaries = bart.get_files("./ImgData/referenceData/cannyDatas")
    testcannyPathDataDictionaries = bart.get_files(cannyPath)
    bart.calculateHu(refcannyDatasDataDictionaries, testcannyPathDataDictionaries, threshold, resultHu2)


if __name__ == '__main__':
    start=time.time()  #开始时间
    test()
    end = time.time()  # 结束时间
    # print(f"整体所使用的时间为: {end - start: .2f} seconds")
    #  调用画框的部分
    from Rectangle import main
    main()
