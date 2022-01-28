# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: discern.py
# @Time    : 2022/1/27 16:07

import cv2
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from torchvision import transforms


def discern(file):
    '''
    测试入口: 手写图片的测试
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('./models/model-mnist.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式

    # 读取要预测的图片
    img = cv2.imread(file)
    # img = cv2.imread(file)
    img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)
    plt.imshow(img, cmap="gray")  # 显示图片
    plt.axis('off')  # 不显示坐标轴

    # 导入图片，图片扩展后为[1，1，32，32]
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为mnist数据集都是灰度图
    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]

    # 预测
    output = model(img)
    prob = F.softmax(output, dim=1)  # prob是10个分类的概率
    print("概率：", prob)
    value, predicted = torch.max(output.data, 1)
    predict = output.argmax(dim=1)
    print("预测类别：", predict.item())
    classes = ['༠', '༡', '༢', '༣', '༤', '༥', '༦', '༧', '༨', '༩']
    return classes[predict.item()]
