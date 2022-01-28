# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: main.py
# @Time    : 2022/1/27 15:22
from matplotlib import pyplot as plt
import time
from train import train_runner
import lenet_5
import dataSet
from inference import test_runner


def main():
    epoch = 100000
    Loss = []
    Accuracy = []
    for epoch in range(1, epoch + 1):
        print("开始时间", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        # 训练部分调用函数
        loss, acc = train_runner(lenet_5.model, lenet_5.device, dataSet.trainloader, lenet_5.optimizer, epoch, Loss,
                                 Accuracy)
        Loss.append(loss)
        Accuracy.append(acc)

        # 测试部分调用函数
        test_runner(lenet_5.device, dataSet.testloader)
        print("结束时间: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')

    print('训练结束啦!')
    plt.subplot(2, 1, 1)
    plt.plot(Loss)
    plt.title('Loss')
    plt.show()
    plt.subplot(2, 1, 2)
    plt.plot(Accuracy)
    plt.title('Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
