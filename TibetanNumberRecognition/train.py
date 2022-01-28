# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: train.py
# @Time    : 2022/1/27 15:20
import torch.nn.functional as F
import torch

"""
定义训练过程
"""
def train_runner(model, device, trainloader, optimizer, epoch,Loss,Accuracy):
    # 训练模型, 启用 BatchNormalization 和 Dropout, 将BatchNormalization和Dropout置为True
    model.train()
    total = 0
    correct = 0.0
    # enumerate迭代已加载的数据集,同时获取数据和数据下标
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # 把模型部署到device上
        inputs, labels = inputs.to(device), labels.to(device)
        # 初始化梯度
        optimizer.zero_grad()
        # 保存训练结果
        outputs = model(inputs)
        # 计算损失和
        # 多分类情况通常使用cross_entropy(交叉熵损失函数), 而对于二分类问题, 通常使用sigmod
        loss = F.cross_entropy(outputs, labels)
        # 获取最大概率的预测结果
        # dim=1表示返回每一行的最大值对应的列下标
        predict = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        if i % 1000 == 0:
            # loss.item()表示当前loss的数值
            print(
                "Train Epoch{} \t Loss: {:.6f}, accuracy: {:.6f}%".format(epoch, loss.item(), 100 * (correct / total)))
            Loss.append(loss.item())
            Accuracy.append(correct / total)
    print(model)
    torch.save(model, './models/model-mnist.pth')  # 保存模型
    return loss.item(), correct / total