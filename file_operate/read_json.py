# _*_ coding:utf-8 _*_
# 103中山分队
# 开发人员：Miao
# 开发时间：2019/12/816:29
# 文件名：read_json.py
# 开发工具：PyCharm
# 功能：读取存储字典的json文件

import json
import numpy as np


def run(file_name):
    # 测试分区拆分
    with open(file_name, mode='r', encoding='gbk') as f2:  # 读取文件中的分区字典
        z_list = json.load(f2)
    return z_list


if __name__ == '__main__':
    # 测试
    # data = run('test_dict2json.json')
    # data = run('E:/博士论文试验数据/chapter4/1606020923/usv_position.json')[:5]
    data = run('E:/博士论文试验数据/chapter5/论文采用数据/50time.json')
    data_2 = run('E:/博士论文试验数据/chapter5/论文采用数据/50time_noise.json')
    # print(data)
    # print(data_2)
    data_3 = np.zeros(len(data))
    for i in range(len(data)):
        data_3[i] = data_2[2] - data[i]
    # print(data_3)
    print(sum(data_3)/len(data_3))
    print(min(data_3))
    print(max(data_3))


