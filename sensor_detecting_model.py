# _*_ coding:utf-8 _*_
# 103中山分队
# 开发人员：runlong
# 开发时间：2019/6/114:04
# 文件名：sensor_detecting_model.py
# 开发工具：PyCharm
# 功能：USV平台探测模型


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def detecting_probability(x0, y0, theta_s, a, b, c, d, r, r_max, alpha):
    """
    计算栅格的探测概率
    :param alpha: 传感器性能参数
    :param r_max: 传感器最大探测距离
    :param x0: USV平台在x轴方向上的坐标位置
    :param y0: USV在y轴方向上的坐标位置
    :param theta_s: USV艏向角
    :param a: 栅格在x轴正方向上的最小值
    :param b: 栅格在x轴负方向上的最大值
    :param c: 栅格在y轴正方向上的最小值
    :param d: 栅格在y轴负方向上的最大值
    :param r: 栅格分辨率
    :return: 栅格的探测概率
    """
    x = np.arange(a, b, r)    # x轴方向划分
    y = np.arange(c, d, r)    # y轴方向划分
    x, y = np.meshgrid(x, y)  # 划分网格
    n = len(x)
    d = np.zeros(shape=(n, n))
    p = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = np.sqrt((x[i, j] - x0) ** 2 + (y[i, j] - y0) ** 2)  # 计算两点间距离
            theta = np.arctan2((y[i, j] - y0), (x[i, j] - x0)) - theta_s  # 计算栅格点与USV平台连线
            p[i, j] = alpha * np.cos(theta) * np.exp(-(d[i, j] / r_max) ** 2)

    return p


# 测试代码
def main():

    # 传感器性能参数
    r_max = 5  # 最大探测距离
    alpha = 0.9

    # 任务区域设定
    a = -5  # x轴正向边界
    b = 5  # x轴负向边界
    c = 0  # y轴正向边界
    d = 10  # y轴负向边界
    r = 0.1  # 栅格分辨率
    x0 = 0  # USV在x轴方向上位置坐标
    y0 = 0  # USV在y轴方向上位置坐标
    theta_s = 0.5 * np.pi  # USV艏向

    # 绘制三维图
    fig = plt.figure()
    ax = Axes3D(fig)

    # 划分网格
    x = np.arange(a, b, r)    # x轴方向划分
    y = np.arange(c, d, r)    # y轴方向划分
    x, y = np.meshgrid(x, y)  # 划分网格

    # 求各个网格对应的探测概率
    z = detecting_probability(x0, y0, theta_s, a, b, c, d, r, r_max, alpha)

    # 画图
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # 设置z轴
    ax.set_zlim(0, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # # 设置色谱带
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__ == "__main__":
    main()
