# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: TransitionCsv.py
# @Time    : 2021/11/14 09:06

import os
import pandas as pd
'''
功能：将excel转换为csv文件
'''
def readFile(filePath):
    fileDictionaries = dict()
    for index, file in enumerate(os.listdir(filePath)):
        fileDictionaries[file] = filePath
    return fileDictionaries

def transition_csv(fileDictionaries,csvPath):
    for name,path in fileDictionaries.items():
        data=pd.read_excel(path+"/"+name,"Sheet1",index_col=0)
        data.to_csv(csvPath+"/"+name.split(".")[0]+".csv",encoding="utf-8")

if __name__ == '__main__':
    # excel文件路径
    excelPath="./excelFile/nothorse"
    # 转化后存储csv的文件路径
    csvPath="./result/csvFile/nothorse"
    transition_csv(readFile(excelPath),csvPath)
