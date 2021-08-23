# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：20/11/2720:52
# 文件名：location_v4.py
# 开发工具：PyCharm
# 功能：去掉重复度版本


import numpy as np
from file_operate import makedir, list2json, read_json
import time
from itertools import combinations
from copy import deepcopy
import math
import pandas as pd


def generate_data(u_num, a_position, s_path):
    """
    生成usv位置，真实时差，噪声，带噪声时差数据
    :param s_path: 数据保存路径
    :param u_num: usv的数目
    :param a_position: auv的位置
    :return:
    """
    c_speed = 1.5  # 水声速度，km/s

    # 1.随机生成USV位置，并保存
    usv_position = np.zeros((u_num, 3))
    row, col = usv_position.shape
    for i in range(row):
        for j in range(col - 1):
            usv_position[i][j] = np.random.uniform(-10, 10)
    list2json.run(usv_position, str(u_num)+'usv_position.json', s_path)

    # 2.计算各个USV距离AUV的距离和AUV向USV发送水声信号耗费时间，并保存
    d = np.zeros(u_num)  # 各个USV与AUV间的真实距离
    c_t = np.zeros(u_num)  # 水声信号以直线形式从AUV发送到USV所耗费时间的真实值（真实时差）
    c_t_n = np.zeros(u_num)  # 带噪声的时差信息
    t_sigma = np.zeros(u_num)  # 噪声标准差
    for i in range(u_num):
        d[i] = np.sqrt((usv_position[i][0] - a_position[0]) ** 2
                       + (usv_position[i][1] - a_position[1]) ** 2 + a_position[2] ** 2)
        c_t[i] = d[i] / c_speed
    list2json.run(c_t, str(u_num)+'time.json', s_path)

    # 3.根据距离为各个时差加入噪声
    for i in range(u_num):
        if d[i] > 9:
            t_sigma[i] = np.random.uniform(0.09, 0.1)           # 随机生成噪声方差
            c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)   # 以一定方差生成时差噪声
            while c_t_n[i] < c_t[i]:  # 由于水声通信存在延时，带噪声的测量值一定大于真实值
                t_sigma[i] = np.random.uniform(0.09, 0.1)
                c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)
        elif 9 > d[i] >= 8:
            t_sigma[i] = np.random.uniform(0.08, 0.09)           # 随机生成噪声方差
            c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)   # 以一定方差生成时差噪声
            while c_t_n[i] < c_t[i]:  # 由于水声通信存在延时，带噪声的测量值一定大于真实值
                t_sigma[i] = np.random.uniform(0.08, 0.09)
                c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)
        elif 8 > d[i] >= 7:
            t_sigma[i] = np.random.uniform(0.07, 0.08)           # 随机生成噪声方差
            c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)   # 以一定方差生成时差噪声
            while c_t_n[i] < c_t[i]:  # 由于水声通信存在延时，带噪声的测量值一定大于真实值
                t_sigma[i] = np.random.uniform(0.07, 0.08)
                c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)
        elif 7 > d[i] >= 6:
            t_sigma[i] = np.random.uniform(0.06, 0.07)           # 随机生成噪声方差
            c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)   # 以一定方差生成时差噪声
            while c_t_n[i] < c_t[i]:  # 由于水声通信存在延时，带噪声的测量值一定大于真实值
                t_sigma[i] = np.random.uniform(0.06, 0.07)
                c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)
        elif 6 > d[i] >= 5:
            t_sigma[i] = np.random.uniform(0.05, 0.06)           # 随机生成噪声方差
            c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)   # 以一定方差生成时差噪声
            while c_t_n[i] < c_t[i]:  # 由于水声通信存在延时，带噪声的测量值一定大于真实值
                t_sigma[i] = np.random.uniform(0.05, 0.06)
                c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)
        elif 5 > d[i] >= 4:
            t_sigma[i] = np.random.uniform(0.04, 0.05)           # 随机生成噪声方差
            c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)   # 以一定方差生成时差噪声
            while c_t_n[i] < c_t[i]:  # 由于水声通信存在延时，带噪声的测量值一定大于真实值
                t_sigma[i] = np.random.uniform(0.04, 0.05)
                c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)
        elif 4 > d[i] >= 3:
            t_sigma[i] = np.random.uniform(0.03, 0.04)           # 随机生成噪声方差
            c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)   # 以一定方差生成时差噪声
            while c_t_n[i] < c_t[i]:  # 由于水声通信存在延时，带噪声的测量值一定大于真实值
                t_sigma[i] = np.random.uniform(0.03, 0.04)
                c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)
        elif 3 > d[i] >= 2:
            t_sigma[i] = np.random.uniform(0.02, 0.03)           # 随机生成噪声方差
            c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)   # 以一定方差生成时差噪声
            while c_t_n[i] < c_t[i]:  # 由于水声通信存在延时，带噪声的测量值一定大于真实值
                t_sigma[i] = np.random.uniform(0.02, 0.03)
                c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)
        elif 2 > d[i] >= 1:
            t_sigma[i] = np.random.uniform(0.01, 0.02)           # 随机生成噪声方差
            c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)   # 以一定方差生成时差噪声
            while c_t_n[i] < c_t[i]:  # 由于水声通信存在延时，带噪声的测量值一定大于真实值
                t_sigma[i] = np.random.uniform(0.01, 0.02)
                c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)
        elif 1 > d[i] >= 0:
            t_sigma[i] = np.random.uniform(0.001, 0.01)           # 随机生成噪声方差
            c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)   # 以一定方差生成时差噪声
            while c_t_n[i] < c_t[i]:  # 由于水声通信存在延时，带噪声的测量值一定大于真实值
                t_sigma[i] = np.random.uniform(0.001, 0.01)
                c_t_n[i] = c_t[i] + t_sigma[i] * np.random.randn(1)
    list2json.run(c_t_n, str(u_num)+'time_noise.json', s_path)
    list2json.run(t_sigma, str(u_num)+'t_sigma.json', s_path)
    return usv_position, c_t, c_t_n, t_sigma


def confidence_matrix_1(u_position, c_t_n, t_sigma, s_path, beta_l=0.3, beta_m=0.5, beta_u=0.8, s_threshold_rate=0.09):
    """
    求解支持度矩阵
    :param u_position: usv位置
    :param c_t_n: 带噪声的时差数据
    :param t_sigma: 噪声标准差
    :param s_threshold_rate: 最佳支持度阈值比例
    :param beta_l: 支持度函数参数下限
    :param beta_m: 支持度函数参数中值
    :param beta_u: 支持度函数参数上限
    :param s_path: 保存路径
    :param threshold_step: 支持度函数阈值步长
    :param threshold_min: 支持度函数阈值最小值
    :return:统计支持度矩阵和最佳融合形式的组合列表
    """
    c_speed = 1.5  # 水声速度，km/s

    # 1.随机选择三组时差数据，估计AUV位置
    usv_list = range(len(u_position))
    combine_list = combine(usv_list, 3)
    combine_num = len(combine_list)
    position_est = np.zeros((combine_num, 2))  # 位置估计
    for i in range(combine_num):
        usv1 = combine_list[i][0]  # 组合中第一个usv编号
        usv2 = combine_list[i][1]  # 组合中第二个usv编号
        usv3 = combine_list[i][2]  # 组合中第三个usv编号

        x1, y1 = u_position[usv1][0], u_position[usv1][1]  # 第一个usv坐标值
        x2, y2 = u_position[usv2][0], u_position[usv2][1]  # 第二个usv坐标值
        x3, y3 = u_position[usv3][0], u_position[usv3][1]  # 第三个usv坐标值

        t1 = c_t_n[usv1]  # 第一个usv获得带噪声的测量时差
        t2 = c_t_n[usv2]  # 第二个usv获得带噪声的测量时差
        t3 = c_t_n[usv3]  # 第三个usv获得带噪声的测量时差

        # 利用三边定位法求解AUV位置
        k0 = 2 * (x1 - x3) * (y2 - y3) - 2 * (x2 - x3) * (y1 - y3)
        k1 = x1**2 - x3**2 + y1**2 - y3**2 + c_speed**2 * t3**2 - c_speed**2 * t1**2
        k2 = x2 ** 2 - x3 ** 2 + y2 ** 2 - y3 ** 2 + c_speed ** 2 * t3 ** 2 - c_speed ** 2 * t2 ** 2

        position_est[i, 0] = (k1 * (y2 - y3) + k2 * (y3 - y1)) / k0  # x轴坐标
        position_est[i, 1] = (k1 * (x3 - x2) + k2 * (x1 - x3)) / k0  # y轴坐标
    # 2.置信距离矩阵
    confidence_m = np.zeros((combine_num, combine_num))  # 置信距离矩阵，此矩阵为对称矩阵
    for i in range(combine_num):
        for j in range(combine_num):
            # 参与位置估算的标准差
            sig_i_0 = t_sigma[combine_list[i][0]]
            sig_i_1 = t_sigma[combine_list[i][1]]
            sig_i_2 = t_sigma[combine_list[i][2]]
            sig2_i = sig_i_0 ** 2 + sig_i_1 ** 2 + sig_i_2 ** 2  # 方差加和
            sig_j_0 = t_sigma[combine_list[j][0]]
            sig_j_1 = t_sigma[combine_list[j][1]]
            sig_j_2 = t_sigma[combine_list[j][2]]
            sig2_j = sig_j_0 ** 2 + sig_j_1 ** 2 + sig_j_2 ** 2  # 方差加和

            if j > i:  # 上三角区域
                confidence_m[i][j] = np.sqrt((position_est[i][0] - position_est[j][0]) ** 2
                                             + (position_est[i][1] - position_est[j][0]) ** 2) / (sig2_i + sig2_j)
            elif j < i:  # 下三角区域
                confidence_m[i][j] = confidence_m[j][i]
    # 3.一致性矩阵
    support_m = np.zeros((combine_num, combine_num))  # 支持度矩阵，此矩阵为对称矩阵
    for i in range(combine_num):
        for j in range(combine_num):
            support_m[i][j] = support_func(confidence_m[i][j], beta_l, beta_m, beta_u)
    # 4.最佳融合置信距离矩阵
    sum_in_row = sum(np.asarray(support_m).T)  # 各行加和
    row_num = []                               # 存储超过支持度阈值的行号
    while True:
        for i in range(len(sum_in_row)):
            if sum_in_row[i] > s_threshold_rate * combine_num:
                row_num.append(i)
        confidence_m_best = np.zeros((len(row_num), len(row_num)))  # 最佳融合置信距离矩阵
        combine_best = []                                           # 最佳融合的组合形式
        position_est_best = []
        for i in range(len(row_num)):
            for j in range(len(row_num)):
                confidence_m_best[i][j] = confidence_m[row_num[i]][row_num[j]]
            combine_best.append(combine_list[row_num[i]])
            position_est_best.append(position_est[row_num[i]])
        if len(confidence_m_best) != 0:
            list2json.run(np.array(combine_best), str(len(u_position))+'_'+str(s_threshold_rate) + 'usv_combination.json', s_path)
            list2json.run(np.array(position_est_best), str(len(u_position))+'_'+str(s_threshold_rate) + 'auv_position_est.json', s_path)
            print('s_threshold_rate is:')
            print(s_threshold_rate)
            break
        else:
            s_threshold_rate = s_threshold_rate/2
    # 5.统计支持度矩阵
    support_m_final = np.ones((len(row_num), len(row_num)))  # 统计支持度矩阵
    max_d = max(map(max, confidence_m_best))                 # 最佳融合置信距离矩阵中的最大值
    for i in range(len(row_num)):
        for j in range(len(row_num)):
            if j > i:
                support_m_final[i][j] = (max_d - confidence_m_best[i][j]) / max_d
            elif j < i:
                support_m_final[i][j] = support_m_final[j][i]
    return support_m_final, combine_best, position_est_best, position_est


def confidence_matrix_2(a_position, u_position, c_t_n, t_sigma, s_path, beta_l=0.3, beta_m=0.5, beta_u=0.8,
                        s_threshold_rate=0.09):
    """
    求解支持度矩阵(考虑重复输入数据)
    :param a_position: auv位置
    :param u_position: usv位置
    :param c_t_n: 带噪声的时差数据
    :param t_sigma: 噪声标准差
    :param s_threshold_rate: 最佳支持度阈值比例
    :param beta_l: 支持度函数参数下限
    :param beta_m: 支持度函数参数中值
    :param beta_u: 支持度函数参数上限
    :param s_path: 保存路径
    :param threshold_step: 支持度函数阈值步长
    :param threshold_min: 支持度函数阈值最小值
    :return:统计支持度矩阵和最佳融合形式的组合列表
    """
    c_speed = 1.5  # 水声速度，km/s

    # 1.随机选择三组时差数据，估计AUV位置
    usv_list = range(len(u_position))
    combine_list = combine(usv_list, 3)
    list2json.run(np.array(combine_list), 'combination_list.json', s_path)
    combine_num = len(combine_list)
    position_est = np.zeros((combine_num, 2))  # 位置估计
    for i in range(combine_num):
        usv1 = combine_list[i][0]  # 组合中第一个usv编号
        usv2 = combine_list[i][1]  # 组合中第二个usv编号
        usv3 = combine_list[i][2]  # 组合中第三个usv编号

        x1, y1 = u_position[usv1][0], u_position[usv1][1]  # 第一个usv坐标值
        x2, y2 = u_position[usv2][0], u_position[usv2][1]  # 第二个usv坐标值
        x3, y3 = u_position[usv3][0], u_position[usv3][1]  # 第三个usv坐标值

        t1 = c_t_n[usv1]  # 第一个usv获得带噪声的测量时差
        t2 = c_t_n[usv2]  # 第二个usv获得带噪声的测量时差
        t3 = c_t_n[usv3]  # 第三个usv获得带噪声的测量时差

        # 利用三边定位法求解AUV位置
        k0 = 2 * (x1 - x3) * (y2 - y3) - 2 * (x2 - x3) * (y1 - y3)
        k1 = x1**2 - x3**2 + y1**2 - y3**2 + c_speed**2 * t3**2 - c_speed**2 * t1**2
        k2 = x2 ** 2 - x3 ** 2 + y2 ** 2 - y3 ** 2 + c_speed ** 2 * t3 ** 2 - c_speed ** 2 * t2 ** 2

        position_est[i, 0] = (k1 * (y2 - y3) + k2 * (y3 - y1)) / k0  # x轴坐标
        position_est[i, 1] = (k1 * (x3 - x2) + k2 * (x1 - x3)) / k0  # y轴坐标
    # 2.置信距离矩阵
    confidence_m_x = np.zeros((combine_num, combine_num))  # x轴方向上置信距离矩阵，此矩阵为对称矩阵
    confidence_m_y = np.zeros((combine_num, combine_num))  # y轴方向上置信距离矩阵，此矩阵为对称矩阵
    for i in range(combine_num):
        for j in range(combine_num):
            # 参与位置估算的标准差
            sig_i_0 = t_sigma[combine_list[i][0]]
            sig_i_1 = t_sigma[combine_list[i][1]]
            sig_i_2 = t_sigma[combine_list[i][2]]
            sig2_i = sig_i_0 ** 2 + sig_i_1 ** 2 + sig_i_2 ** 2  # 方差加和
            sig_j_0 = t_sigma[combine_list[j][0]]
            sig_j_1 = t_sigma[combine_list[j][1]]
            sig_j_2 = t_sigma[combine_list[j][2]]
            sig2_j = sig_j_0 ** 2 + sig_j_1 ** 2 + sig_j_2 ** 2  # 方差加和

            if j > i:  # 上三角区域
                same = [x for x in combine_list[i] if x in combine_list[j]]
                dif1 = [x for x in combine_list[i] if x not in combine_list[j]]
                dif2 = [x for x in combine_list[j] if x not in combine_list[i]]
                # 相同输入数据为1个
                if len(same) == 1:
                    u1 = u_position[dif1[0]]
                    u2 = u_position[dif2[1]]
                    u3 = u_position[same[0]]
                    u4 = u_position[dif2[0]]
                    u5 = u_position[dif2[1]]

                    t1 = t_sigma[dif1[0]]
                    t2 = t_sigma[dif1[1]]
                    t3 = t_sigma[same[0]]
                    t4 = t_sigma[dif2[0]]
                    t5 = t_sigma[dif2[1]]
                    [b_x_1, b_y_1] = repeatability_value_1(a_position, u1, u2, u3, u4, u5, t1, t2, t3, t4,
                                                           t5)
                    confidence_m_x[i][j] = np.sqrt((position_est[i][0] - position_est[j][0]) ** 2
                                                   + (position_est[i][1] - position_est[j][0]) ** 2) / ((1 - 0.5*b_x_1) * (
                            sig2_i + sig2_j))
                    confidence_m_y[i][j] = np.sqrt((position_est[i][0] - position_est[j][0]) ** 2
                                                   + (position_est[i][1] - position_est[j][0]) ** 2) / ((1 - 0.5*b_y_1) * (
                            sig2_i + sig2_j))
                # 相同输入数据为2个
                elif len(same) == 2:
                    u1 = u_position[dif1[0]]
                    u2 = u_position[same[0]]
                    u3 = u_position[same[1]]
                    u4 = u_position[dif2[0]]

                    t1 = t_sigma[dif1[0]]
                    t2 = t_sigma[same[0]]
                    t3 = t_sigma[same[1]]
                    t4 = t_sigma[dif2[0]]

                    [b_x_2, b_y_2] = repeatability_value_2(a_position, u1, u2, u3, u4, t1, t2, t3, t4)
                    confidence_m_x[i][j] = np.sqrt((position_est[i][0] - position_est[j][0]) ** 2
                                                   + (position_est[i][1] - position_est[j][0]) ** 2) / (
                                                       (1 - 0.5 * b_x_2) * (
                                                       sig2_i + sig2_j))
                    confidence_m_y[i][j] = np.sqrt((position_est[i][0] - position_est[j][0]) ** 2
                                                   + (position_est[i][1] - position_est[j][0]) ** 2) / (
                                                       (1 - 0.5 * b_y_2) * (
                                                       sig2_i + sig2_j))
                # 无相同输入
                elif len(same) == 0:
                    confidence_m_x[i][j] = np.sqrt((position_est[i][0] - position_est[j][0]) ** 2
                                                   + (position_est[i][1] - position_est[j][0]) ** 2) / (
                                                       sig2_i + sig2_j)
                    confidence_m_y[i][j] = np.sqrt((position_est[i][0] - position_est[j][0]) ** 2
                                                   + (position_est[i][1] - position_est[j][0]) ** 2) / (
                                                       sig2_i + sig2_j)
            elif j < i:  # 下三角区域
                confidence_m_x[i][j] = confidence_m_x[j][i]
                confidence_m_y[i][j] = confidence_m_y[j][i]
    # 3.一致性矩阵
    support_m_x = np.zeros((combine_num, combine_num))  # 支持度矩阵，此矩阵为对称矩阵
    support_m_y = np.zeros((combine_num, combine_num))  # 支持度矩阵，此矩阵为对称矩阵
    for i in range(combine_num):
        for j in range(combine_num):
            support_m_x[i][j] = support_func(confidence_m_x[i][j], beta_l, beta_m, beta_u)
            support_m_y[i][j] = support_func(confidence_m_y[i][j], beta_l, beta_m, beta_u)
    # 4.最佳融合置信距离矩阵，x轴和y轴方向上同时满足大于设定阈值
    sum_in_row_x = sum(np.asarray(support_m_x).T)  # 各行加和
    row_num_x = []                                 # 存储超过支持度阈值的行号
    sum_in_row_y = sum(np.asarray(support_m_x).T)  # 各行加和
    row_num_y = []                                 # 存储超过支持度阈值的行号

    while True:
        for i in range(len(sum_in_row_x)):
            if sum_in_row_x[i] > s_threshold_rate * combine_num and sum_in_row_y[i] > s_threshold_rate * combine_num:
                row_num_x.append(i)
                row_num_y.append(i)
        confidence_m_best_x = np.zeros((len(row_num_x), len(row_num_x)))  # 最佳融合置信距离矩阵
        combine_best_x = []                                               # 最佳融合的组合形式
        position_est_best_x = []                                          # 最佳融合列表中的位置在x轴方向上的估计值
        confidence_m_best_y = np.zeros((len(row_num_x), len(row_num_x)))  # 最佳融合置信距离矩阵
        combine_best_y = []                                               # 最佳融合的组合形式
        position_est_best_y = []                                          # 最佳融合列表中的位置在y轴方向上的估计值
        for i in range(len(row_num_x)):
            for j in range(len(row_num_x)):
                confidence_m_best_x[i][j] = confidence_m_x[row_num_x[i]][row_num_x[j]]
                confidence_m_best_y[i][j] = confidence_m_y[row_num_y[i]][row_num_y[j]]
            combine_best_x.append(combine_list[row_num_x[i]])
            position_est_best_x.append(position_est[row_num_x[i]])
            combine_best_y.append(combine_list[row_num_y[i]])
            position_est_best_y.append(position_est[row_num_y[i]])
        if len(confidence_m_best_x) != 0 and len(confidence_m_best_y) != 0:
            list2json.run(np.array(combine_best_x),
                          str(len(u_position)) + '_' + str(s_threshold_rate) + 'usv_combination.json', s_path)
            list2json.run(np.array(position_est_best_x),
                          str(len(u_position)) + '_' + str(s_threshold_rate) + 'auv_position_est_x.json', s_path)
            list2json.run(np.array(position_est_best_y),
                          str(len(u_position)) + '_' + str(s_threshold_rate) + 'auv_position_est_y.json', s_path)
            print('s_threshold_rate is:')
            print(s_threshold_rate)
            break
        else:
            s_threshold_rate = s_threshold_rate/2

    # 5.统计支持度矩阵
    support_m_final_x = np.ones((len(row_num_x), len(row_num_x)))  # 统计支持度矩阵
    max_d_x = max(map(max, confidence_m_best_x))                   # 最佳融合置信距离矩阵中的最大值
    support_m_final_y = np.ones((len(row_num_y), len(row_num_y)))  # 统计支持度矩阵
    max_d_y = max(map(max, confidence_m_best_y))                   # 最佳融合置信距离矩阵中的最大值
    for i in range(len(row_num_x)):
        for j in range(len(row_num_x)):
            if j > i:
                support_m_final_x[i][j] = (max_d_x - confidence_m_best_x[i][j]) / max_d_x
                support_m_final_y[i][j] = (max_d_y - confidence_m_best_y[i][j]) / max_d_y
            elif j < i:
                support_m_final_x[i][j] = support_m_final_x[j][i]
                support_m_final_y[i][j] = support_m_final_y[j][i]
    list2json.run(support_m_final_x, 'support_m_final_x', s_path)
    list2json.run(support_m_final_y, 'support_m_final_y', s_path)
    return support_m_final_x, combine_best_x, position_est_best_x, support_m_final_y, combine_best_y, position_est_best_y, position_est


def combine(temp_list, n):
    """
    从列表中随机选择n个元素，并将所有组合形式输出
    :param temp_list:列表
    :param n:组合的元素个数
    :return:
    """
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2


def support_func(d, beta_l, beta, beta_u):
    """
    支持度函数
    :param d: 置信度距离
    :param beta_l: 支持度函数下限参数
    :param beta: 支持度函数中值参数
    :param beta_u: 支持度函数上限参数
    :return: 支持度
    """
    if d <= beta_l:
        r = 1
    elif beta_l < d <= beta:
        r = 1 - 0.5 * (d - beta_l) / (beta_l - beta)
    elif beta < d <= beta_u:
        r = 0.5 * (beta_u - d) / (beta_u - beta)
    else:
        r = 0
    return r


def auv_position_est_0(a_position):
    """
    基于加权平均估算auv位置
    :param a_position: 带噪声的auv位置信息
    :return: 融合位置结果
    """
    a_position = np.array(a_position)
    row, col = a_position.shape
    x_total = 0
    y_total = 0
    for i in range(row):
        x_total += a_position[i][0]
        y_total += a_position[i][1]
    [x_position, y_position] = [x_total/row, y_total/row]
    return [x_position, y_position]


def auv_position_est_1(s_m_f, p_est):
    """
    基于支持度的位置融合估计
    :param s_m_f: 统计支持度矩阵
    :param p_est: 最优位置融合估计列表
    :return:auv位置坐标
    """
    # 根据上述统计支持度矩阵估算AUV位置
    row, col = s_m_f.shape
    total_support_value = 0  # 支持度总和
    w = np.zeros(row)        # 每行之和
    for i in range(row):
        for j in range(col):
            if i != j:
                total_support_value += s_m_f[i][j]
                w[i] += s_m_f[i][j]
    if total_support_value != 0:
        w = w / total_support_value  # 每个位置估计权重列表
    position_est = np.array(p_est)
    [auv_x, auv_y] = w @ np.array(position_est)  # auv融合估计结果
    return [auv_x, auv_y]


def auv_position_est_2(theta_x_m, theta_y_m, p_est_x, p_est_y):
    """
    基于支持度的位置融合估计，去除重复度
    :param theta_x_m: x轴方向上的支持度矩阵
    :param theta_y_m: y轴方向上的支持度矩阵
    :param p_est_x: x轴方向上最优位置融合估计列表
    :param p_est_y: y轴方向上最优位置融合估计列表
    :return:auv位置坐标
    """
    # 根据上述统计支持度矩阵估算AUV位置
    row, col = theta_x_m.shape
    total_support_value_x = 0  # x轴坐标位置的支持度总和
    total_support_value_y = 0  # y轴坐标位置的支持度总和
    w_x = np.zeros(row)        # 每行之和
    w_y = np.zeros(row)        # 每行之和
    for i in range(row):
        for j in range(col):
            if i != j:
                total_support_value_x += theta_x_m[i][j]
                total_support_value_y += theta_y_m[i][j]
                w_x[i] += theta_x_m[i][j]
                w_y[i] += theta_y_m[i][j]
    if total_support_value_x != 0:
        w_x = w_x / total_support_value_x  # 每个位置估计权重列表
    if total_support_value_y != 0:
        w_y = w_y / total_support_value_y  # 每个位置估计权重列表
    position_est_x = np.array(p_est_x)
    position_est_y = np.array(p_est_y)
    auv_x = w_x @ np.array(position_est_x[:, 0])  # x轴上auv位置融合估计结果
    auv_y = w_y @ np.array(position_est_y[:, 1])  # y轴上auv位置融合估计结果
    return [auv_x, auv_y]


def sigma_position(p, p_est):
    """
    计算位置标准差
    :param p_est:最终融合位置
    :param p: 位置估计数据
    :return: x轴和y轴上坐标位置的标准差
    """
    p = np.array(p)
    row, col = p.shape
    mean_x = p_est[0]
    mean_y = p_est[1]
    sum_x = 0
    sum_y = 0
    for i in range(row):
        sum_x += (p[i][0] - mean_x)**2
        sum_y += (p[i][1] - mean_y)**2
    sigma_x = np.sqrt(sum_x/row)
    sigma_y = np.sqrt(sum_y/row)
    return [sigma_x, sigma_y]


def a_value(a_position, u_position):
    """
    计算参数a
    :param a_position: 基于运动模型的auv位置预测值
    :param u_position: usv位置
    :return:
    """
    r = np.sqrt((u_position[0] - a_position[0])**2 + (u_position[1] - a_position[1])**2
                + (u_position[2] - a_position[2])**2)
    a_x = (u_position[0] - a_position[0]) / r
    a_y = (u_position[1] - a_position[1]) / r
    a_z = (u_position[2] - a_position[2]) / r
    return [a_x, a_y, a_z]


def repeatability_value_1(a_position, u1_position, u2_position, u3_position, u4_position, u5_position, u1_sigma,
                          u2_sigma, u3_sigma, u4_sigma, u5_sigma):
    """
    计算一个相同输入数据条件下重复度
    :param a_position: 基于运动模型的auv位置预测值
    :param u1_position: usv1的位置
    :param u2_position: usv2的位置
    :param u3_position: usv3的位置
    :param u4_position: usv4的位置
    :param u5_position: usv5的位置
    :param u1_sigma: usv1的时差测量噪声
    :param u2_sigma: usv2的时差测量噪声
    :param u3_sigma: usv3的时差测量噪声，重复数据
    :param u4_sigma: usv4的时差测量噪声
    :param u5_sigma: usv5的时差测量噪声
    :return: 重复度
    """
    c_speed = 1.5  # 水声速度，km/s
    # 计算参数a
    [a_x_1, a_y_1, a_z_1] = a_value(a_position, u1_position)
    [a_x_2, a_y_2, a_z_2] = a_value(a_position, u2_position)
    [a_x_3, a_y_3, a_z_3] = a_value(a_position, u3_position)
    [a_x_4, a_y_4, a_z_4] = a_value(a_position, u4_position)
    [a_x_5, a_y_5, a_z_5] = a_value(a_position, u5_position)

    # 计算参数b
    b0 = a_x_1 * a_y_2 * a_z_3 - a_x_1 * a_y_3 * a_z_2 - a_x_2 * a_y_1 * a_z_3 + a_x_2 * a_y_3 * a_z_1 \
         + a_x_3 * a_y_1 * a_z_2 + a_x_3 * a_y_2 * a_z_1
    b1 = a_x_3 * a_y_4 * a_z_5 - a_x_3 * a_y_5 * a_z_4 - a_x_4 * a_y_3 * a_z_5 + a_x_4 * a_y_5 * a_z_3 \
         + a_x_5 * a_y_3 * a_z_4 + a_x_5 * a_y_4 * a_z_3

    # 计算参数k
    k_x_1 = c_speed * (a_x_2 * a_y_3 - a_x_3 * a_y_2) / b0
    k_x_2 = -1 * c_speed * (a_x_2 * a_z_3 - a_x_3 * a_z_2) / b0
    k_x_3_0 = c_speed * (a_y_2 * a_z_3 - a_y_3 * a_z_2) / b0

    k_y_1 = c_speed * (a_x_1 * a_z_3 - a_x_3 * a_z_1) / b0
    k_y_2 = -1 * c_speed * (a_x_1 * a_y_3 - a_x_3 * a_y_1) / b0
    k_y_3_0 = -1 * c_speed * (a_y_1 * a_z_3 - a_y_3 * a_z_1) / b0

    k_x_3_1 = c_speed * (a_x_4 * a_y_5 - a_x_5 * a_y_4) / b1
    k_x_4 = -1 * c_speed * (a_x_4 * a_z_5 - a_x_5 * a_z_4) / b1
    k_x_5 = c_speed * (a_y_4 * a_z_5 - a_y_5 * a_z_4) / b1

    k_y_3_1 = c_speed * (a_x_3 * a_z_5 - a_x_5 * a_z_3) / b1
    k_y_4 = -1 * c_speed * (a_x_3 * a_y_5 - a_x_5 * a_y_3) / b1
    k_y_5 = -1 * c_speed * (a_y_3 * a_z_5 - a_y_5 * a_z_3) / b1

    # 协方差矩阵
    k_11_x = k_x_1 ** 2 * u1_sigma ** 2 + k_x_2 ** 2 * u2_sigma ** 2 + k_x_3_0 ** 2 * u3_sigma ** 2
    k_12_x = k_x_3_0 ** 2 * k_x_3_1 ** 2 * u3_sigma ** 2
    k_22_x = k_x_3_1 ** 2 * u3_sigma ** 2 + k_x_4 ** 2 * u4_sigma ** 2 + k_x_5 ** 2 * u5_sigma ** 2

    k_11_y = k_y_1 ** 2 * u1_sigma ** 2 + k_y_2 ** 2 * u2_sigma ** 2 + k_y_3_0 ** 2 * u3_sigma ** 2
    k_12_y = k_y_3_0 ** 2 * k_y_3_1 ** 2 * u3_sigma ** 2
    k_22_y = k_y_3_1 ** 2 * u3_sigma ** 2 + k_y_4 ** 2 * u4_sigma ** 2 + k_y_5 ** 2 * u5_sigma ** 2

    # 重复度
    b_x = math.log(2 * np.pi * np.e * k_12_x) / (
                math.log(2 * np.pi * np.e * k_11_x) + math.log(2 * np.pi * np.e * k_22_x))
    if b_x > 1:
        b_x = 1
    elif b_x < 0:
        b_x = 0
    b_y = math.log(2 * np.pi * np.e * k_12_y) / (
                math.log(2 * np.pi * np.e * k_11_y) + math.log(2 * np.pi * np.e * k_22_y))
    if b_y > 1:
        b_y = 1
    elif b_y < 0:
        b_y = 0

    return [b_x, b_y]


def repeatability_value_2(a_position, u1_position, u2_position, u3_position, u4_position, u1_sigma, u2_sigma,
                          u3_sigma, u4_sigma):
    """
    计算两个相同输入数据条件下重复度
    :param a_position: 基于运动模型的auv位置预测值
    :param u1_position: usv1的位置
    :param u2_position: usv2的位置
    :param u3_position: usv3的位置
    :param u4_position: usv4的位置
    :param u1_sigma: usv1的时差测量噪声
    :param u2_sigma: usv2的时差测量噪声，重复数据
    :param u3_sigma: usv3的时差测量噪声，重复数据
    :param u4_sigma: usv4的时差测量噪声
    :return: 重复度
    """
    c_speed = 1.5  # 水声速度，km/s

    # 计算参数a
    [a_x_1, a_y_1, a_z_1] = a_value(a_position, u1_position)
    [a_x_2, a_y_2, a_z_2] = a_value(a_position, u2_position)
    [a_x_3, a_y_3, a_z_3] = a_value(a_position, u3_position)
    [a_x_4, a_y_4, a_z_4] = a_value(a_position, u4_position)

    # 计算参数b
    b0 = a_x_1 * a_y_2 * a_z_3 - a_x_1 * a_y_3 * a_z_2 - a_x_2 * a_y_1 * a_z_3 + a_x_2 * a_y_3 * a_z_1 \
         + a_x_3 * a_y_1 * a_z_2 + a_x_3 * a_y_2 * a_z_1
    b2 = a_x_2 * a_y_3 * a_z_4 - a_x_2 * a_y_4 * a_z_3 - a_x_3 * a_y_2 * a_z_4 + a_x_3 * a_y_4 * a_z_2 \
         + a_x_4 * a_y_2 * a_z_3 + a_x_4 * a_y_3 * a_z_2

    # 计算参数k
    k_x_1 = c_speed * (a_x_2 * a_y_3 - a_x_3 * a_y_2) / b0
    k_x_2_0 = -1 * c_speed * (a_x_2 * a_z_3 - a_x_3 * a_z_2) / b0
    k_x_3_0 = c_speed * (a_y_2 * a_z_3 - a_y_3 * a_z_2) / b0

    k_y_1 = c_speed * (a_x_1 * a_z_3 - a_x_3 * a_z_1) / b0
    k_y_2_0 = -1 * c_speed * (a_x_1 * a_y_3 - a_x_3 * a_y_1) / b0
    k_y_3_0 = -1 * c_speed * (a_y_1 * a_z_3 - a_y_3 * a_z_1) / b0

    k_x_2_1 = c_speed * (a_x_3 * a_y_4 - a_x_4 * a_y_3) / b2
    k_x_3_1 = -1 * c_speed * (a_x_3 * a_z_4 - a_x_4 * a_z_3) / b2
    k_x_4 = c_speed * (a_y_3 * a_z_4 - a_y_4 * a_z_3) / b2

    k_y_2_1 = c_speed * (a_x_2 * a_z_4 - a_x_4 * a_z_2) / b2
    k_y_3_1 = -1 * c_speed * (a_x_2 * a_y_4 - a_x_4 * a_y_2) / b2
    k_y_4 = -1 * c_speed * (a_y_2 * a_z_4 - a_y_4 * a_z_2) / b2

    # 协方差矩阵
    k_11_x = k_x_1 ** 2 * u1_sigma ** 2 + k_x_2_0 ** 2 * u2_sigma ** 2 + k_x_3_0 ** 2 * u3_sigma ** 2
    k_12_x = k_x_2_0 ** 2 * k_x_2_1 ** 2 * u2_sigma ** 2 + k_x_3_0 ** 2 * k_x_3_1 ** 2 * u3_sigma ** 2
    k_22_x = k_x_2_1 ** 2 * u2_sigma ** 2 + k_x_3_1 ** 2 * u3_sigma ** 2 + k_x_4 ** 2 * u4_sigma ** 2

    k_11_y = k_y_1 ** 2 * u1_sigma ** 2 + k_y_2_0 ** 2 * u2_sigma ** 2 + k_y_3_0 ** 2 * u3_sigma ** 2
    k_12_y = k_y_2_0 ** 2 * k_y_2_1 ** 2 * u2_sigma ** 2 + k_y_3_0 ** 2 * k_y_3_1 ** 2 * u3_sigma ** 2
    k_22_y = k_y_2_1 ** 2 * u2_sigma ** 2 + k_y_3_1 ** 2 * u3_sigma ** 2 + k_y_4 ** 2 * u4_sigma ** 2

    # 重复度
    b_x = math.log(2 * np.pi * np.e * k_12_x) / (
                math.log(2 * np.pi * np.e * k_11_x) + math.log(2 * np.pi * np.e * k_22_x))
    if b_x > 1:
        b_x = 1
    elif b_x < 0:
        b_x = 0
    b_y = math.log(2 * np.pi * np.e * k_12_y) / (
                math.log(2 * np.pi * np.e * k_11_y) + math.log(2 * np.pi * np.e * k_22_y))
    if b_y > 1:
        b_y = 1
    elif b_y < 0:
        b_y = 0

    return [b_x, b_y]


def main():
    print(__file__ + " start!!")
    save_file_name = save_path + 'result.csv'
    # 生成新的usv数据
    u_data, time_c, t_data, t_sigma_data = generate_data(usv_num, auv_position, save_path)
    # 读取已有的usv数据
    # u_data = read_json.run('E:/博士论文试验数据/chapter4/1606202166/50usv_position.json')
    # t_data = read_json.run('E:/博士论文试验数据/chapter4/1606202166/50time_noise.json')
    # t_sigma_data = read_json.run('E:/博士论文试验数据/chapter4/1606202166/50t_sigma.json')

    for i in range(5, usv_num+1):
        print('***' * 20)
        print('USV数目为:%d' % i)
        # 导入数据
        # u_data, time_c, t_data, t_sigma_data = generate_data(i, auv_position, save_path)
        usv_position = u_data[:i]
        time_c_n = t_data[:i]
        time_sigma = t_sigma_data[:i]

        # 支持度矩阵计算
        support_m_final, list_combination, position_est_b, position_est_noise \
            = confidence_matrix_1(usv_position, time_c_n, time_sigma, save_path, s_threshold_rate=0.1)

        # 加权平均估算auv位置
        [auv_x_position_0, auv_y_position_0] = auv_position_est_0(position_est_noise)
        [sigma_x_0, sigma_y_0] = sigma_position(position_est_noise, [auv_x_position_0, auv_y_position_0])
        print('加权平均位置估计结果:')
        print([auv_x_position_0, auv_y_position_0])
        print([sigma_x_0, sigma_y_0])

        # 最佳融合估计
        [auv_x_position_1, auv_y_position_1] = auv_position_est_1(support_m_final, position_est_b)
        [sigma_x_1, sigma_y_1] = sigma_position(position_est_b, [auv_x_position_1, auv_y_position_1])
        print("最佳融合估计结果：")
        print([auv_x_position_1, auv_y_position_1])
        print([sigma_x_1, sigma_y_1])

        # 支持度矩阵计算
        support_m_final_x, list_combination_x, position_est_b_x, support_m_final_y, list_combination_y, position_est_b_y\
            , position_est_noise = confidence_matrix_2(auv_position, usv_position, time_c_n, time_sigma, save_path, s_threshold_rate=0.1)

        # 去除重复度的最佳融合估计
        [auv_x_position_2, auv_y_position_2] = auv_position_est_2(support_m_final_x, support_m_final_y, position_est_b_x, position_est_b_y)
        [sigma_x_2, sigma_y_2] = sigma_position(position_est_b, [auv_x_position_2, auv_y_position_2])
        print("去掉重复度的位置估计：")
        print([auv_x_position_2, auv_y_position_2])
        print([sigma_x_2, sigma_y_2])

        data = [i, auv_x_position_0, auv_y_position_0, sigma_x_0, sigma_y_0,
                   auv_x_position_1, auv_y_position_1, sigma_x_1, sigma_y_1,
                   auv_x_position_2, auv_y_position_2, sigma_x_2, sigma_y_2]

        data2csv = pd.DataFrame([data])
        data2csv.to_csv(save_file_name, mode='a', header=False, index=None)


if __name__ == '__main__':
    # 数据保存路径
    start_time = time.time()
    save_path = 'E:/博士论文试验数据/chapter5/' + str(int(start_time)) + '/'
    makedir.mkdir(save_path)

    # usv数目
    usv_num = 50

    # auv位置
    auv_position = [0, 1, -1]
    main()





