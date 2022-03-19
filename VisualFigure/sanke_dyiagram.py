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
