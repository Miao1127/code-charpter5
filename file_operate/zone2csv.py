# _*_ coding:utf-8 _*_
# 103中山分队
# 开发人员：Miao
# 开发时间：2019/12/816:31
# 文件名：zone2csv.py
# 开发工具：PyCharm
# 功能：将每个分区的栅格信息分别写入单独的csv文件

import pandas as pd

data = [1, 2, 3]

zone = pd.DataFrame([data], columns=['x', 'y', 'z'])
zone.to_csv('test.csv', mode='a', index=None, header=False)
zone.to_csv('test.csv', mode='a', index=None, header=False)





