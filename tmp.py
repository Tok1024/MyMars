#导入决策树相关包
import numpy as np
import pandas as pd
import math
from math import log

#创建数据
def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels

#打印数据
datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)

# 计算给定数据集的熵（信息熵）
def calc_ent(datasets):
    # 计算数据集的长度
    data_length = len(datasets)
    # 统计数据集中每个类别的出现次数
    label_count = {}
    for i in range(data_length):
        # 获取每个样本的标签
        label = datasets[i][-1]
        # 如果该类别不在label_count中，则添加到label_count中
        if label not in label_count:
            label_count[label] = 0
        # 统计该类别的出现次数
        label_count[label] += 1
    # 计算熵
    ent = 0
    for count in label_count.values():
        p = count / data_length
        ent -= p * log(p, 2)
    return ent

# 计算信息增益
def info_gain(ent, cond_ent):
    # 信息增益等于熵减去条件熵
    return ent - cond_ent

# 使用信息增益选择最佳特征作为根节点特征进行决策树的训练
def info_gain_train(datasets):
    # 计算特征的数量
    count = len(datasets[0]) - 1
    # 计算整个数据集的熵
    ent = calc_ent(datasets)
    # 存储每个特征的信息增益
    best_feature = []
    for c in range(count):
        # 计算每个特征的条件熵
        c_info_gain = info_gain(ent, cond_ent(datasets, axis=c))
        # 将特征及其对应的信息增益存入best_feature列表中
        best_feature.append((c, c_info_gain))
        # 输出每个特征的信息增益
        print('特征({}) 的信息增益为： {:.3f}'.format(labels[c], c_info_gain))
    # 找到信息增益最大的特征
    best_ = ''' code '''
    # 返回信息增益最大的特征作为根节点特征
    return '特征({})的信息增益最大，选择为根节点特征'.format(labels[best_[0]])