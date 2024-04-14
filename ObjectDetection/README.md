### 目标检测需求
分割与二值化函数放在一起；分割后的Hu距之后，相似度阈值小于0.08的图进入下一个文件夹(每个跟一个确定的参考图进行计算Hu值)；
对感兴趣区域计算边缘图，随后②中确认的参考图进比较，阈值小于0.6的为马；
#### 一般效果
就是在数字后面加一个点，再加一个空格。不过看起来起来可能不够明显。    
ImgData为数据，代码以及数据结构具体如下：

1. ObjectDetection      整个项目文件夹
2. ImgData          数据集
3. referenceData        参考数据集文件
4. BinaryDataSets   分割并二值化后的文件夹
5. cannyDatas       对感兴图像进行边缘后的文件夹
6. InterestedData   分割并二值化后提取感兴趣后的文件夹
7. sourceData       原始数据集
8. testData             测试数据集文件
9. BinaryDataSets   分割并二值化后的文件夹
10. cannyDatas       对感兴图像进行边缘后的文件夹
11. InterestedData   分割并二值化后提取感兴趣后的文件夹
12. sourceData       原始数据集
13. resultHu1        第一次匹配结果集
14. resultHu2        第二次匹配结果集
15. Models                 分割模型存储
16. Result                  结果存储集
17. ReadMe.docx             说明文件
18. testUnit.py          测试脚本
19. trainUnit.py         训练脚本
20. unit.py              各函数集
