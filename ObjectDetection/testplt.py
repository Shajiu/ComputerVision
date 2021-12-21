# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: testplt.py
# @Time    : 2021/11/21 12:32
import matplotlib.pyplot as plt
import numpy as np

# plt.switch_backend('agg')
def function():
    # 生成数据
    x = np.arange(0, 10, 0.1) # 横坐标数据为从0到10之间，步长为0.1的等差数组
    y = np.sin(x) # 纵坐标数据为 x 对应的 sin(x) 值
    # 生成图形
    plt.plot(x, y)
    # 显示图形
    plt.show()

if __name__ == '__main__':
    function()
    print("运行成功!!!")