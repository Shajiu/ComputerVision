# -*- coding: utf-8 -*-
# @Author  : Shajiu
# @FileName: sankeDyiagram.py
# @Time    : 2022/3/19 16:56
import pandas as pd
from pyecharts.charts import Sankey
from pyecharts import options as opts

df = pd.DataFrame({
    '性别': ['男', '男', '男', '女', '女', '女'],
    '熬夜原因': ['单身', '脱单', '未知', '单身', '脱单', '未知'],
    "文化程度":["小学","初中","大专","小学","初中","大专"],
    '人数': [57, 13, 30, 33, 5, 62]
})
print(df)

nodes = []

# for i in range(2):
for i in range(3):
    values = df.iloc[:, i].unique()
    for value in values:
        dic = {}
        dic['name'] = value
        nodes.append(dic)
print(nodes)

first=df.groupby(["性别","熬夜原因"])["人数"].sum().reset_index()
second=df.iloc[:,1:]
first.columns=["source","target","value"]
second.columns=["source","target","value"]
result=pd.concat([first,second])
result.head(10)

linkes = []

# for i in df.values:

for i in result.values:
    dic = {}
    dic['source'] = i[0]
    dic['target'] = i[1]
    dic['value'] = i[2]
    linkes.append(dic)

print(linkes)

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
pic.render('sankeDyiagram.html')
