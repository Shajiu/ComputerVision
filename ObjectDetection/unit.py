# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: unit.py
# @Time    : 2021/11/7 09:39
import os
import cv2
from pixellib.semantic import semantic_segmentation
import codecs
import time

class ObjectDetection(object):
    def __init__(self):
        pass

    def get_files(self,filePath):
        '''
        遍历文件路径下的文件
        :return:   字典型{文件名称，绝对路径}
        '''
        fileDictionaries=dict()
        for index, file in enumerate(os.listdir(filePath)):
            fileDictionaries[file]=filePath
        return fileDictionaries

    def semanticBinaryzation(self,semanticModel,dataBinaryzation,PathDictionaries):
        '''
        分割并二值化图像
        :param semanticModel:        分割模型路径
        :param dataBinaryzation:     分割并二值化后存储的路径
        :param PathDictionaries:     原始文件 字典型
        :return: 存储分割并二值化后的文件
        '''
        segment_image = semantic_segmentation()
        segment_image.load_pascalvoc_model(semanticModel)
        start = time.time()  # 开始时间
        for name,path in PathDictionaries.items():
            segment_image.segmentAsPascalvoc(os.path.join(path,name), output_image_name =os.path.join(dataBinaryzation,name))
            img = cv2.imread(os.path.join(dataBinaryzation,name))
            Grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(Grayimg, 12, 255,cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(dataBinaryzation,name), thresh)
        end=time.time()  # 结束时间
        print(f"分割并二值化图像使用的时间为: {end - start: .2f} seconds","\t数据大小为:",len(PathDictionaries))

    def extraction(self,sourcePathDictionaries,binaryzationPathDictionaries,featureImagePath):
        '''
        针对二值化后的图像提取感兴趣的区域
        :param sourcePathDictionaries:           原始的图像路径
        :param binaryzationPathDictionaries:    分割并二值化后的图像路径
        :param featureImagePath:                提取感兴趣之后的存储路径
        :return:                                保存感兴趣的图像
        '''
        start = time.time()  # 开始时间
        for name,filepath in binaryzationPathDictionaries.items():
           assert name in sourcePathDictionaries
           im1 = cv2.imread(sourcePathDictionaries[name]+"/"+name)
           im2_quyu = cv2.imread(filepath+"/"+name)
           img_masked = cv2.bitwise_and(im2_quyu, im1)
           cv2.imwrite(featureImagePath+"/"+name, img_masked)
        end=time.time()
        print(f"针对二值化后的图像提取感兴趣的区域时间为:{end - start: .2f} seconds","\t数据大小为:",len(sourcePathDictionaries))
    def canny(self,featureImagePathPathDictionaries,cannyPath):
        '''
        提取边缘
        :param featureImagePathPathDictionaries:     针对感兴趣的图像进行提取边缘
        :param cannyPath:                            存储边缘后的路径
        :return:                                     存储图像边缘后的文件
        '''
        start=time.time()  # 开始时间
        for name,filePath in featureImagePathPathDictionaries.items():
            image = cv2.imread(os.path.join(filePath,name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(image, 50, 230)
            cv2.imwrite(os.path.join(cannyPath,name),canny)
        end=time.time() #结束时间
        print(f"调用针对感兴趣区域进行边缘提取的函数时间:{end - start: .2f} seconds","\t数据大小为:",len(featureImagePathPathDictionaries))
    def calculateHu(self,referencePath, dataBinaryzation, threshold,resultPath):
        '''
        计算参考文件与待检测图像之间的Hu值
        :param referencePath:       二值化/边缘后的参考文件路径
        :param fileDictionaries:   待检测图像二值化/边缘后的文件路径
        :return:
        '''
        start = time.time()  # 记录开始时间
        huMatchResult=dict()
        result=codecs.open(os.path.join("./result",str(threshold)+".txt"),'w',encoding="utf-8")
        for refName,refPath in referencePath.items():
            IM1 = cv2.imread(os.path.join(refPath,refName), cv2.IMREAD_GRAYSCALE)
            for name,filepath in dataBinaryzation.items():
                IM2 = cv2.imread(os.path.join(filepath,name), cv2.IMREAD_GRAYSCALE)
                matchScore= cv2.matchShapes(IM1, IM2, cv2.CONTOURS_MATCH_I2, 0)
                if matchScore<threshold:
                    huMatchResult[name]=[matchScore,filepath]
                    cv2.imwrite(os.path.join(resultPath,name), IM2)
                    res="参考图"+refName+"与测试图"+name+"之间的Hu值为:"+str(matchScore)
                    result.write(res+"\n")
        end = time.time()  # 结束时间
        print(f"计算参考文件与待检测图像之间的Hu值所用的时间:{end - start: .2f} seconds","\t数据大小为:",len(dataBinaryzation))
        return huMatchResult