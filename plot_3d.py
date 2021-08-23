# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：20/12/1019:24
# 文件名：plot_3d.py
# 开发工具：PyCharm
# 功能：绘制三维散点图


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from file_operate import read_json
import numpy as np


usv_position = read_json.run('E:/博士论文试验数据/chapter5/1606483803/50usv_position.json')
usv_position = np.array(usv_position)
auv_position = [0, 0, -1]
fig = plt.figure(figsize=(15, 13))
axes3d = Axes3D(fig)
# 设置坐标格式
plt.tick_params(labelsize=25)
labels = axes3d.get_xticklabels() + axes3d.get_yticklabels() + axes3d.get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
axes3d.set_xlabel('x(km)', fontdict={'family': 'Times New Roman', 'style': 'italic', 'size': 25}, labelpad=15)
axes3d.set_ylabel('y(km)', fontdict={'family': 'Times New Roman', 'style': 'italic', 'size': 25}, labelpad=15)
axes3d.set_zlabel('z(km)', fontdict={'family': 'Times New Roman', 'style': 'italic', 'size': 25}, labelpad=15)
axes3d.set_zlim3d(0, 12)

# 绘制USV位置
axes3d.scatter(usv_position[:, 0], usv_position[:, 1], usv_position[:, 2], s=60, marker='^', c='r')
# # 绘制AUV真实位置
axes3d.scatter(auv_position[0], auv_position[1], auv_position[2], s=60, marker='o', c='r')
axes3d.text(auv_position[0], auv_position[1], auv_position[2], 'AUV', fontsize=25, family='Times New Roman')
# 绘制AUV估算位置
# axes3d.scatter(auv_x, auv_y, auv_position[2], s=50, marker='^', c='r')

plt.grid(True)
# 设置坐标轴刻度
axes3d.set_xlim3d(-10, 10)
axes3d.set_ylim3d(-10, 10)
axes3d.set_zlim3d(-1, 1)
plt.show()
