# -*-coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import signal
import image_analysis as im
import sys


def kuramoto_2D(theta, K, Omega, Kernel, edge=False):
    # K 結合定数.
    Ctheta = np.cos(theta)
    Stheta = np.sin(theta)
    CthetaC = signal.convolve2d(Ctheta, Kernel, 'same', 'symm')
    SthetaC = signal.convolve2d(Stheta, Kernel, 'same', 'symm')
    # fill pad input arrays with fillvalue. (default)
    # wrap circular boundary conditions.
    # symm symmetrical boundary conditions.
    if edge is True:
        for i in np.arange(int(Kernel.shape[0] / 2)):
            CthetaC[i, :] = signal.convolve2d(Ctheta, Kernel, 'same')
    new_theta = Omega + K * (Ctheta * SthetaC - Stheta * CthetaC)
    return new_theta


def Runge_Kutta4(theta, K, Omega, h, Kernel):  # Runge-Kutta h:time step
    k1 = kuramoto_2D(theta, K, Omega, Kernel)
    k2 = kuramoto_2D(theta + 0.5 * h * k1, K, Omega, Kernel)
    k3 = kuramoto_2D(theta + 0.5 * h * k2, K, Omega, Kernel)
    k4 = kuramoto_2D(theta + h * k3, K, Omega, Kernel)
    y = theta + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def gaussian_kernel(ksize, sigma=False):  # ガウシアン行列作る．
    if sigma is False:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        print('SIGMA = ' + str(sigma))
    kernel_1d = np.empty((ksize, 1), dtype=np.float64)  # 1次元で作って拡張
    for i in np.arange(ksize):
        kernel_1d[i] = np.exp(-1 * np.power(i - (ksize - 1) * 0.5, 2) / (2 * np.power(sigma, 2)))
    kernel_2d = kernel_1d.dot(kernel_1d.T)
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    print(kernel_2d)


if __name__ == '__main__':
    gaussian_kernel(ksize=5, sigma=False)
    sys.exit()
    roop = 1000  # ループ回数
    h = 0.1  # 時間刻み
    R = 250  # 解析のサイズ
    C = 150  # 解析のサイズ
    # 細胞サイズを20μ，フロンドサイズを3mm * 5mm とした時．
    K = 0.01  # k/n
    Omega = 1 + 0.01 * np.random.randn(R, C)  # 角速度
    theta = np.empty((roop, R, C))
    theta[0] = np.random.rand(R, C) * 2 * np.pi  # 初期位相
    # theta[0] = np.random.rand(R,C) * 0.5
    # Omega = 1 + 0.3 * np.random.randn(R,C)# 角速度
    # t = np.array((0,h,h*roop))

    area = 9  # 相互作用が何個隣まであるか
    Kernel = np.ones((1 + area * 2, 1 + area * 2))
    for i in np.arange(1, roop):
        theta[i] = Runge_Kutta4(theta[i - 1], K, Omega, h, Kernel)

    print(np.max(theta))
    theta = np.mod(theta, 2 * np.pi) / (2 * np.pi)
    print(np.max(theta))

    theta_color = np.empty((theta.shape[0], theta.shape[1], theta.shape[2], 3), dtype=np.int16)
    for i in np.arange(roop):
        theta_color[i] = im.make_color(theta[i])

    im.save_imgs('LLLL_symm_area-9_k-0.01', theta_color)
