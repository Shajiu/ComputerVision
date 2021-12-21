# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: Analysis.py
# @Time    : 2021/11/14 09:18

import csv
import codecs
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.switch_backend("agg")

def readFile(filePath):
    '''
    :param filePath:   待读取文件的路径
    :return: {"文件名称"：“文件路径”,,,,,}
    '''
    fileDictionaries = dict()
    for index, file in enumerate(os.listdir(filePath)):
        fileDictionaries[file] = filePath
    return fileDictionaries

def read_csv(pathDictionaries):
    '''
    :param pathDictionaries: {"文件名称"：“文件路径”,,,,,}
    :return:
    '''
    cList=[]  # 存储整个文件夹下的C列
    dList=[]  # 存储整个文件夹下的D列
    XList=[]  # 存储整个文件夹下的C/D列对应的(A,B)，即对应的坐标点
    count=0   # 统计整个文件夹下读取的文件数目
    Rows=[]           # 存储整个文件夹下的D列中大于300的占比值；
    Cmax = []         # 存储整个文件夹下的C列中最大值；
    nameList = []     # 存储整个文件夹下每个文件名；
    CountRows=[]      # 存储大于(D>300 and C＜60)的个数

    nameSource=[]     # 存储大于(D>300 and C＜60)的文件名称
    for name,path in pathDictionaries.items():
        count += 1  # 累计
        # 读取对应的文件
        csv_reader=csv.reader(codecs.open(path+"//"+name))
        Row=0     # 累计每个文件中的行数
        row=0     # 累计每个文件中满足条件的行数(D＞300 and C＜60)
        c_x=[]    # 存储每个文件中的C列数
        d_x=[]    # 存储每个文件中的D列数
        xy=[]     # 存储每个文件中的A,B列数（坐标(X,Y)=(A,B)）
        #   读取每个文件中的每一行数据
        for line in csv_reader:
            rowX=str(line[0])           # x坐标
            columnY=str(line[1])        # y坐标
            xy.append("(" + rowX + "," + columnY + ")")      # 构造坐标
            XList.append("(" + rowX + "," + columnY + ")")   # 构造坐标
            cList.append(math.fabs(float(line[2])))
            c_x.append(math.fabs(float(line[2])))
            dList.append(math.fabs(float(line[3])))
            d_x.append(math.fabs(float(line[3])))
            if (math.fabs(float(line[3])) > 300):
                row+=1  # 累计每个文件中满足条件的行数(D＞300 and C＜60)
            Row+=1
        Rows.append(float(row/Row))  #  D列中大于300的占比
        Cmax.append(max(c_x))     #  C列中最大值
        nameList.append(name)
        CountRows.append(row)
        paint(c_x,d_x,xy,name)  #调用画图函数

        #  统计符合的文件名
        if row>6:
            nameSource.append(name.split(".")[0])


    write_cvs(nameList,CountRows,Rows,Cmax)  # 调用编写结果文件
    write_result(nameSource)                 # 编写最筛选出的结果


def paint(y1,y2,X,name):
    fig = plt.figure(figsize=(15, 10))  # 设置画布大小
    plt.tick_params(axis='x', labelsize=8)  # 设置x轴标签大小
    plt.xticks(rotation=-45)  # 设置x轴标签旋转角度
    l1 = plt.plot(X, y1, 'r-*', label='C-column')
    l2 = plt.plot(X, y2, 'g-*', label='D-column')
    #plt.title('The Lasers in Three Conditions')
    plt.xlabel('Coordinate')
    plt.ylabel('C/D')
    plt.legend()
    plt.savefig('./result/horse_jpg/' + name.split(".")[0] + ".jpg")

def write_cvs(nameList,CountRows,Rows,Cmax) :
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'文件名称': nameList,"D大于300的且C小于60的个数":CountRows,"D大于300的且C小于60的占比":Rows,"C的均值":Cmax})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("./result/horse_csv/horse.csv", index=False, sep=',',encoding="utf-8")
    # 每张图中D大于300且C小于60的个数、C的均值、文件名称、最终结果命名
    paint(CountRows,Cmax, nameList, "最终结果.")


def write_result(nameSource):
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'文件名称': nameSource})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("./result/最终筛选出的结果.csv", index=False, sep=',', encoding="utf-8")

if __name__ == '__main__':
    csvPath = "./result/csvFile/horse"
    read_csv(readFile(csvPath))
