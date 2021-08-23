# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：20/11/1913:44
# 文件名：list2json.py
# 开发工具：PyCharm
# 功能：将列表保存到json文件中


import json
import os
import numpy as np


def run(list_name, json_file_name, json_file_save_path):
    """
    将list写入到json文件
    :param list_name:
    :param json_file_name: 写入的json文件名字
    :param json_file_save_path: json文件存储路径
    :return:
    """
    os.chdir(json_file_save_path)
    save_list = list_name.tolist()
    with open(json_file_name, 'w') as f:
        json.dump(save_list, f)


if __name__ == '__main__':
    a = np.ones((10, 2))
    run(a, 'a.json', './')
