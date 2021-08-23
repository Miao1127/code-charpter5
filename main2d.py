# _*_ coding:utf-8 _*_
# 开发人员：103中山分队-苗润龙
# 开发时间：20/11/2416:22
# 文件名：main2d.py.py
# 开发工具：PyCharm
# 功能：2d显示效果

import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from file_operate import makedir
import time
from solver import support_matrix_final


def calc_input():
    """
    计算输入量，设置运动质点的线速度和角速度
    :return:
    """
    v = 1  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation(xTrue, xd, u):
    """
    观测器
    :param xTrue: 质点状态真实值
    :param xd: 质点状态推算值
    :param u: 输入量
    :return: 质点状态真实值、带噪声的GPS观测值、带噪声控制量下的质点状态推算值、带噪声的控制量
    """
    xTrue = motion_model(xTrue, u)  # 质点运动模型，输入参数为质点状态和控制量

    # 带噪声的GPS观测数据
    # z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # 生成数据
    usv_position, time_c, time_c_n, time_sigma = support_matrix_final.generate_data(usv_num, [xTrue[0], xTrue[1], -1], save_path)

    # 计算统计支持度矩阵
    support_m_final, list_combination, position_est_b, position_est \
        = support_matrix_final.confidence_matrix(usv_position, time_c_n, time_sigma, save_path)

    # 根据上述统计支持度矩阵估算AUV位置
    [auv_x, auv_y] = support_matrix_final.auv_position_est_2(support_m_final, position_est_b, list_combination,
                                                [xTrue[0], xTrue[1], -1], usv_position, time_sigma)
    z = [[auv_x], [auv_y]]
    # 向控制量中加入噪声
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)  # 带噪声控制量下质点状态推算值

    return xTrue, z, xd, ud


def motion_model(x, u):
    """
    AUV运动模型
    :param x: 状态值
    :param u: 控制量
    :return: 新的状态值
    """
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[dt * math.cos(x[2, 0]), 0],
                  [dt * math.sin(x[2, 0]), 0],
                  [0.0, dt],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def usv_motion_model(x, u):
    """
    USV运动模型
    :param x: 状态值
    :param u: 控制量
    :return: 新的状态值
    """
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[dt * math.cos(x[2, 0]), 0],
                  [dt * math.sin(x[2, 0]), 0],
                  [0.0, dt],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


# 观测模型
def observation_model(x):
    """
    观测模型
    :param x: 状态值
    :return: 观测值
    """
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def jacob_f(x, u):
    """
    运动模型雅克比矩阵

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -dt * v * math.sin(yaw), dt * math.cos(yaw)],
        [0.0, 1.0, dt * v * math.cos(yaw), dt * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h():
    # 观测模型雅克比矩阵
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u):
    #  预测
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    #  更新
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    """
    绘制方差椭圆
    :param xEst: 质点状态估计值
    :param PEst: 方差矩阵
    :return:
    """
    Pxy = PEst[0:2, 0:2]                 # 质点状态中x和y轴方向上位置坐标的方差矩阵
    eigval, eigvec = np.linalg.eig(Pxy)  # 计算矩阵的特征值和特征向量

    # 将较大的特征值排在第一位
    if eigval[0] >= eigval[1]:  # 第一特征值大于等于第二特征值
        bigind = 0
        smallind = 1
    else:                       # 第二特征值大于第一特征值
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])            # 椭圆长边
    # b = math.sqrt(eigval[smallind])          # 椭圆短边
    b = a
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    n = len(px)
    pz = np.zeros(n)
    plt.plot(px, py, pz, "--r")


def main():
    print(__file__ + " start!!")

    time = 0.0

    # AUV状态向量 [x y yaw v]'
    xEst = np.zeros((4, 1))   # 状态估计值
    xTrue = np.zeros((4, 1))  # 状态真实值
    PEst = np.eye(4)          # 方差矩阵

    xDR = np.zeros((4, 1))    # 状态推算值

    # 历史数据
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    fig = plt.figure(figsize=(15, 13))

    while True:
        time += dt
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            # plot_covariance_ellipse(xEst, PEst)
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    # 数据保存路径
    start_time = time.time()
    save_path = 'E:/博士论文试验数据/chapter5/' + str(int(start_time)) + '/'
    save_path = str(save_path)
    makedir.mkdir(save_path)

    # EKF方差矩阵，预测状态方差和观测状态方差
    Q = np.diag([
        0.001,  # x轴位置方差
        0.001,  # y轴位置方差
        np.deg2rad(1.0),  # 艏向角方差
        1.0  # 速度方差
    ]) ** 2  # 预测状态方差
    R = np.diag([0.001, 0.001]) ** 2  # 观测方差矩阵

    #  仿真参数设置
    INPUT_NOISE = np.diag([0.001, np.deg2rad(30.0)]) ** 2  # 控制量噪声
    GPS_NOISE = np.diag([0.005, 0.005]) ** 2               # GPS噪声方差矩阵
    usv_num = 10                                           # usv数目
    auv_position = [0, 0, -1]                               # auv初始位置
    c_speed = 1.5                                          # 水下声速，km/s
    dt = 0.1  # 时间步长

    show_animation = True
    main()


