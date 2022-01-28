#### 藏语手写字母与PyQt的碰撞，基于PyTorch框架LeNet-5网络的图形化藏文手写字母识别系统So Easy
##### 引言：
&ensp;&ensp;今天我通过PyTorch框架，使用LeNet-5网络实现手写藏文字母的识别系统。为了能够更好友好利用，通过PyQt分别实现了上传图片和在线绘制的应用软件，在藏语字母数据集上训练分类器可以看作是图像识别的“Hello world”，感兴趣的朋友可以一起来聊聊呀！
准备数据：
&ensp;&ensp;本次数据集是本人最近两年每次回家过春节期间在家收集的手写藏文字母数据。包含67864张手写藏文字母图像:其中55000张用于训练，12864张用于测试。图像是灰度的，28x28像素的，并且居中的，以减少预处理和加快运行。
![图1 手写藏文字母数据集](https://s3.bmp.ovh/imgs/2022/01/1ffee1905380446c.jpg)
&ensp;由于手写藏文字母数据集图片尺寸是 28x28 单通道的，而LeNet-5 网络输入 Input 图片尺寸是 32x32，因此使用 Transforms.Resize 将输入图片尺寸调整为 32x32。
##### 数据转换:
&ensp; &ensp;原始图像分别放到train和test文件夹中，注意默认格式为PNG，若是JPG要修改代码，而且图像要按照标签存放。
##### 设置环境：
- Python==3.7.3；
- Tensorflow==1.14.0；
- Pytorch==1.8.0+cpu;
- OpenCv==3.4.2;
- PIL==8.0.1;
- Transformers==2.5.1;
- PyQt5==5.15.4
##### 构建网络
&ensp;&ensp; 这里要解释一下Pytorch 手写藏文字母数据集标准化为什么是transforms.Normalize((0.1307,), (0.3081,))？标准化（Normalization）是神经网络对数据的一种经常性操作。标准化处理指的是：样本减去它的均值，再除以它的标准差，最终样本将呈现均值为0方差为1的数据分布。
&ensp;&ensp;神经网络模型偏爱标准化数据，原因是均值为0方差为1的数据在sigmoid、tanh经过激活函数后求导得到的导数很大，反之原始数据不仅分布不均（噪声大）而且数值通常都很大（本例中数值范围是0~255），激活函数后求导得到的导数则接近与0，这也被称为梯度消失。前文已经分析，神经网络是根据函数对权值求导的导数来调整权值，导数越大，调整幅度越大，越快逼近目标函数，反之，导数越小，调整幅度越小，所以说，数据的标准化有利于加快神经网络的训练。
&ensp;&ensp;除此之外，还需要保持train_set、val_set和test_set标准化系数的一致性。标准化系数就是计算要用到的均值和标准差，在本例中是((0.1307,), (0.3081,))，均值是0.1307，标准差是0.3081，这些系数都是数据集提供方计算好的数据。不同数据集就有不同的标准化系数，例如([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])就是ImageNet dataset的标准化系数（RGB三个通道对应三组系数），当需要将Imagenet预训练的参数迁移到另一神经网络时，被迁移的神经网络就需要使用Imagenet的系数，否则预训练不仅无法起到应有的作用甚至还会帮倒忙，例如，我们想要用神经网络来识别夜空中的星星，因为黑色是夜空的主旋律，从像素上看黑色就是数据集的均值，标准化操作时，所有图像会减去均值（黑色），如此Imagenet预训练的神经网络很难识别出这些数据是夜空图像！
##### 训练步骤
- 第一步：加载数据，并做出一定的预先处理dataSet.py：
``` Python
# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: dataSet.py
# @Time    : 2022/1/27 15:11
import torch
from torchvision import datasets, transforms
"""
使用LeNet-5网络结构创建藏文数据识别分类器
"""
pipline_train = transforms.Compose([
    #随机旋转图片
    transforms.RandomHorizontalFlip(),
    #将图片尺寸resize到32x32
    transforms.Resize((32,32)),
    #将图片转化为Tensor格式
    transforms.ToTensor(),
    #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
    transforms.Normalize((0.1307,),(0.3081,))
])
pipline_test = transforms.Compose([
    #将图片尺寸resize到32x32
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])
#下载/读取数据集
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=pipline_train)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=pipline_test)
#加载数据集
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
print("数据加载完毕")
 ```
- 第二步：搭建LeNet-5神经网络结构，并定义前向传播的过程lenet_5.py：
``` Python
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
 ```
- 第三步：将定义好的网络结构搭载到GPU/CPU，并定义优化器：
``` Python
 #创建模型，部署gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
 #定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
- 第四步：定义训练过程：
``` Python
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
 ```
- 第五步：定义测试过程：
``` Python
 # -*- coding: utf-8 -*-
 # @Author  : Shajiu
 # @FileName: inference.py
 # @Time    : 2022/1/27 15:21
import torch
import torch.nn.functional as F
def test_runner(device, testloader):
    #模型验证, 必须要写, 否则只要有输入数据, 即使不训练, 它也会改变权值
    #因为调用eval()将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
    model=torch.load("./models/model-mnist.pth")
    model.eval()
    #统计模型正确率, 设置初始值
    correct = 0.0
    test_loss = 0.0
    total = 0
    #torch.no_grad将不会计算梯度, 也不会进行反向传播
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, label).item()
            predict = output.argmax(dim=1)
            #计算正确数量
            total += label.size(0)
            correct += (predict == label).sum().item()
        #计算损失值
        print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss/total, 100*(correct/total)))
 ```
- 第六步：运行：
``` Python
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
 ```
- 第七步：保存模型：
``` Python
torch.save(model, './models/model-mnist.pth')  # 保存模型
 ```
- 第八步：手写图片的测试：
``` Python
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
    return predict.item()
 ```
##### 分析结果
&ensp;&ensp;经历 400 次 epoch 的 loss 和 accuracy 曲线如下：
![图2 epoch=400时的Loss值](https://s3.bmp.ovh/imgs/2022/01/ac09b849ae52c0e9.png)
![图3 epoch=400时的Accuracy值](https://s3.bmp.ovh/imgs/2022/01/ac498d19b95e361c.png)
&ensp;&ensp;最终在 12864张测试样本上，Average_loss降到了 0.00018，accuracy 达到了 98.12%。可以说 LeNet-5 的效果非常好！
##### PyQt5识别效果
&ensp;&ensp;首先配置PyQt中QT Designer，创建两个Button对象，分别为“选择图片”、“识别结果”，然后创建两个Label对象，分别用于显示相机原图和显示检测后图像。
创建多线程检测机制，分别给两个Button设置不同的槽函数，分别用于触发相机拍照和调用检测函数。运行uploadApp.py可得到如下结果。
![图4 上传图像界面](https://s3.bmp.ovh/imgs/2022/01/a7322898495917b4.jpg)
![图5 上传图像界面识别](https://s3.bmp.ovh/imgs/2022/01/70f6dbffca4a13c4.jpg)
![图6 手写文字界面](https://s3.bmp.ovh/imgs/2022/01/a08e0e44c398e06e.jpg)
&ensp;&ensp;全文的代码都是可以顺利运行的，建议大家自己跑一边。源码、数据、模型都公开，若你想获取源码，请点击[这里](https://github.com/Shajiu/ComputerVision/tree/main/TibetanNumberRecognition)即可。
