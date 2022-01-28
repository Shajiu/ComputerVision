### 项目介绍：
#### 藏语手写字母与PyQt的碰撞，基于PyTorch框架LeNet-5网络的图形化藏文手写字母识别系统So Easy
#### 设计环境
- Python==3.7.3；
- Tensorflow==1.14.0；
- Pytorch==1.8.0+cpu;
- OpenCv==3.4.2;
- PIL==8.0.1;
- Transformers==2.5.1;
- PyQt5==5.15.4
#### 项目简介
- data  此文件夹下含有藏文字母图像转换为二进制的数据集
- models 此文件夹下含有经过10轮的训练模型
- convert-images-to-mnist-format.py  将png格式的数据转换为二进制个数据函数
- dataSet.py   数据加载模块
- discern.py  检测识别模块
- image_jpg_png.py  将jpg文件转换为png文件
- inference.py  推理模块
- lenet_5.py  搭建LeNet-5神经网络结构，并定义前向传播的过程
- main.py   模型训练的主函数入口
- train.py  定义训练过程
- uploadApp.py  上传图像检测识别界面
- writeApp.py   手写识别检测界面
#### 使用说明
- [请可以按照微信公众号上的文章进行操作](https://mp.weixin.qq.com/s/b2fxFJSHG7_O-ZLy--WH_g)
- [请可以按照博客上的文章进行操作](https://shajiu.github.io/2022/01/28/cang-yu-shou-xie-zi-mu-yu-pyqt-de-peng-zhuang-ji-yu-pytorch-kuang-jia-lenet-5-wang-luo-de-tu-xing-hua-cang-wen-shou-xie-zi-mu-shi-bie-xi-tong-so-easy/)
#### 运行writeApp.py可得到如下结果
![图6 手写文字界面](https://s3.bmp.ovh/imgs/2022/01/a08e0e44c398e06e.jpg)
