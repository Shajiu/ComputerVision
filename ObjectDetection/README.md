### 目标检测需求
分割与二值化函数放在一起；
分割后的Hu距之后，相似度阈值小于0.08的图进入下一个文件夹(每个跟一个确定的参考图进行计算Hu值)；
对感兴趣区域计算边缘图，随后②中确认的参考图进比较，阈值小于0.6的为马；
ImgData为数据源
* 代码以及数据结构具体如下：
- ObjectDetection      整个项目文件夹
- ImgData          数据集
- referenceData        参考数据集文件
- BinaryDataSets   分割并二值化后的文件夹
- cannyDatas       对感兴图像进行边缘后的文件夹
- InterestedData   分割并二值化后提取感兴趣后的文件夹
-	sourceData       原始数据集
-	testData             测试数据集文件
- BinaryDataSets   分割并二值化后的文件夹
- cannyDatas       对感兴图像进行边缘后的文件夹
- InterestedData   分割并二值化后提取感兴趣后的文件夹
-	sourceData       原始数据集
-	resultHu1        第一次匹配结果集
-	resultHu2        第二次匹配结果集
-	Models                 分割模型存储
- Result                  结果存储集
-	ReadMe.docx             说明文件
-	testUnit.py          测试脚本
-	trainUnit.py         训练脚本
-	unit.py              各函数集
