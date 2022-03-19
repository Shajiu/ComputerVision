### 利用Python绘制桑基图

&emsp; 桑基图(Sankey diagram)，即桑基能量分流图，也叫桑基能量平衡图。它是一种特定类型的流程图，图中延伸的分支的宽度对应数据流量的大小，通常应用于能源、材料成分、金融等数据的可视化分析。因1898年Matthew Henry Phineas Riall Sankey绘制的"蒸汽机的能源效率图"而闻名，此后便以其名字命名为"桑基图"。

### 汇总提炼
> 桑基两个字取自“发明”者的名字
> 属于流程图的一种，核心在于展示数据的流转
> 主要由节点、边和流量三要素构成，边越宽代表流量越大
> 遵循守恒定律，无论怎么流动，开端和末端数据总是一致的

### Python手把手绘制桑基图
> 动手之前，我们再次敲黑板，回顾桑基图组成要素的重点——节点、边和流量。
> 任何桑基图，无论展现形式如何夸张，色彩如何艳丽，动效如何炫酷，本质都逃不出上述3点。
> 只要我们定义好上述3个要素，Python的pyecharts库能够轻松实现桑基图的绘制。

### 数据实例

|性别 |熬夜原因 |文化程度 | 人数|
|----|----|----|----|
|0|  男   |单身  | 小学 | 57|
|1  |男  | 脱单   |初中  |13|
|2 | 男  | 未知   |大专  |30|
|3 | 女   |单身   |小学  |33|
|4 | 女   |脱单  | 初中   |5|
|5  |女   |未知   |大专  |62|

### Python具体步骤
- 读取数据
```python
df = pd.DataFrame({
    '性别': ['男', '男', '男', '女', '女', '女'],
    '熬夜原因': ['单身', '脱单', '未知', '单身', '脱单', '未知'],
    "文化程度":["小学","初中","大专","小学","初中","大专"],
    '人数': [57, 13, 30, 33, 5, 62]
})
print(df)
 ```
- 首先是节点，这一步需要把所有涉及到的节点去重规整在一起。也就是要把性别一列的“男”、“女”和单身原因一列的“单生”、“脱单”、“未知”以列表内嵌套字典的形式去重汇总：
``` python
for i in range(3):
    values = df.iloc[:, i].unique()
    for value in values:
        dic = {}
        dic['name'] = value
        nodes.append(dic)
print(nodes)
 ```
```
[{'name': '男'}, {'name': '女'}, {'name': '单身'}, {'name': '脱单'}, {'name': '未知'}, {'name': '小学'}, {'name': '初中'}, {'name': '大专'}]
 ```
- 接着，定义边和流量，数据从哪里流向哪里，流量（值）是多少，循环+字典依然可以轻松搞定：
``` Python
first=df.groupby(["性别","熬夜原因"])["人数"].sum().reset_index()
second=df.iloc[:,1:]
first.columns=["source","target","value"]
second.columns=["source","target","value"]
result=pd.concat([first,second])
result.head(10)
linkes = []
for i in result.values:
    dic = {}
    dic['source'] = i[0]
    dic['target'] = i[1]
    dic['value'] = i[2]
    linkes.append(dic)
print(linkes)
 ```
```
[{'source': '女', 'target': '单身', 'value': 33}, {'source': '女', 'target': '未知', 'value': 62}, {'source': '女', 'target': '脱单', 'value': 5}, {'source': '男', 'target': '单身', 'value': 57}, {'source': '男', 'target': '未知', 'value': 30}, {'source': '男', 'target': '脱单', 'value': 13}, {'source': '单身', 'target': '小学', 'value': 57}, {'source': '脱单', 'target': '初中', 'value': 13}, {'source': '未知', 'target': '大专', 'value': 30}, {'source': '单身', 'target': '小学', 'value': 33}, {'source': '脱单', 'target': '初中', 'value': 5}, {'source': '未知', 'target': '大专', 'value': 62}]
 ```
- source-target-value的字典格式，很清晰的描述了数据的流转情况。
``` python
pic = (
    Sankey()
        .add('',  # 图例名称
             nodes,  # 传入节点数据
             linkes,  # 传入边和流量数据
             linestyle_opt=opts.LineStyleOpts(opacity=0.3, curve=0.5, color="source"),  # 设置透明度、弯曲度、颜色
             label_opts=opts.LabelOpts(position="right"),  # 标签显示位置
             node_gap=30,  # 节点之前的距离
             # orient="vertical"  # 要垂直显示
             )
        .set_global_opts(title_opts=opts.TitleOpts(title='桑基图Demo'))
)
pic.render('test.html')
 ```
![桑基图Demo](https://s3.bmp.ovh/imgs/2022/03/1e7db4cc9bf418da.png)
#### Python画境外新冠病例输入桑基图
- 数据，请您点击[这里](https://github.com/Shajiu/ComputerVision/blob/main/VisualFigure/demo.xlsx)获取相应excel数据
- 源码，请您直接复制如下代码即可复制到Pycharm等IDE中即可运行
``` python
# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: sanke_dyiagram.py
# @Time    : 2022/3/19 18:22
import pandas
from pyecharts.charts import Sankey
from pyecharts import options as opts
data = pandas.read_excel("E://demo.xlsx", sheet_name='Sheet1')
nodes = []
for i in set(pandas.concat([data.来源地, data.输出地])):
    d1 = {}
    d1["name"] = i
    nodes.append(d1)
links = []
for x, y, z in zip(data.来源地, data.输出地, data.数量):
    d2 = {}
    d2['source'] = x
    d2['target'] = y
    d2['value'] = z
    links.append(d2)
pic = (
    Sankey(init_opts=opts.InitOpts(width="1600px", height="800px")).add("确诊病例",  # 图例名称
                                                                        nodes,  # 传入节点数据
                                                                        links,  # 传入边和流量数据
                                                                        # 设置透明度、弯曲度、颜色
                                                                        linestyle_opt=opts.LineStyleOpts(opacity=0.5,
                                                                                                         curve=0.5,
                                                                                                         color="source"),
                                                                        # 标签显示位置 right/top
                                                                        label_opts=opts.LabelOpts(position="right"),
                                                                        node_gap=10,  # 节点之前的距离
                                                                        # orient="vertical"  # 垂直显示
                                                                        ).set_global_opts(
        title_opts=opts.TitleOpts(title="TOP10境外输入统计"))
)
pic.render("TOP10境外输入统计.html")
```
- 效果图
![TOP10境外输入统计.html](https://s3.bmp.ovh/imgs/2022/03/dea4de11bda4c9dc.png)

####  若您想获取以上两个Demo的全部源码和数据，请您点击[这里]()即可获取。
