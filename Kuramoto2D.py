# -*-coding: utf-8 -*-
"""
    Twist chain oscillator simulations.

It is based on Poincare oscillators.
Same model as proposed by {Myung, Jihwan, et al. "The choroid plexus is an important circadian clock component." Nature communications 9.1 (2018): 1062.}
"""

import sys

import numpy as np
from scipy import signal


def oscillator_2D(x, y, kernel, omega, A=1):
    """Calculate fx and fy for Runge-Kutta.

    $fx = -r(r-A) - omega * y + SIGMA _i!=j K_(ij) x_(j)$

    Args:
        x: array of x coordinates
        y: array of y coordinates
        kernel: [description]
        omega: angular velocity
        A: Standard state amplitude (default: {1})

    Returns:
        fx (= dx/dt), fy (= dy/dt)
    """
    # K 結合定数.
    r = np.sqrt(np.square(x) * np.square(y))
    field_x = signal.convolve2d(x, kernel, mode="same", boundary="fill", fillvalue=0)
    field_y = signal.convolve2d(y, kernel, mode="same", boundary="fill", fillvalue=0)
    fx = r * (A - r) - omega * y + field_x
    fy = r * (A - r) + omega * x + field_y
    return fx, fy


def RK4(x, y, kernel, omega, A, h):
    """Solve oscillator_2D by Runge–Kutta method.

    Args:
        x: array of x coordinates
        y: array of y coordinates
        kernel: [description]
        omega: angular velocity
        A: Standard state amplitude
        h: Step width

    Returns:
        new_x, new_y
    """
    k1_x, k1_y = oscillator_2D(x, y, kernel, omega, A)
    k2_x, k2_y = oscillator_2D(x + k1_x * 0.5 * h, y + k1_y * 0.5 * h, kernel, omega, A)
    k3_x, k3_y = oscillator_2D(x + k2_x * 0.5 * h, y + k2_y * 0.5 * h, kernel, omega, A)
    k4_x, k4_y = oscillator_2D(x + k3_x * h, y + k3_y * h, kernel, omega, A)
    new_x, new_y = x + h / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x), y + h / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
    return new_x, new_y


def oscillator_2D_RK4(x, y, kernel, omega, A, h, step_count, save_step):
    """"Solve oscillator_2D by Runge–Kutta method.

    Args:
        x: array of x coordinates
        y: array of y coordinates
        kernel: [description]
        omega: angular velocity
        A: Standard state amplitude
        h: Step width
        step_count: Number of calculations
        save_step: Storage frequency(Save all if one.)

    Returns:
        xs, ys
    """
    r, c = x.shape
    save_idx = range(0, step_count, save_step)
    save = 0
    xs = np.empty((len(save_idx), r, c), dtype=np.float64)
    ys = np.empty_like(xs)
    for i in range(step_count):
        x, y = RK4(x, y, kernel, omega, A, h)
        if save_idx[save] == i:
            xs[save], ys[save] = x, y
            save = save + 1
    return xs, ys

if __name__ == '__main__':
    step_count = 1000  # ループ回数
    h = 0.1  # 時間刻み huer
    R = 250  # 解析のサイズ
    C = 150  # 解析のサイズ
    # 細胞サイズを20μ，フロンドサイズを3mm * 5mm とした時．
    A = 1  # amplitude
    omega = 2 * np.pi / 24  # + 0.01 * np.random.randn(R, C)  # 角速度
    theta = np.random.rand(R, C) * 2 * np.pi  # 初期位相
    x = np.cos(theta)
    y = np.sin(theta)
    kernel = np.ones((3, 3)) / 9
    kernel[1, 1] = 0

    oscillator_2D_RK4(x, y, kernel, omega, A, h, step_count, save_step=1)
    # theta_color = np.empty((theta.shape[0], theta.shape[1], theta.shape[2], 3), dtype=np.int16)
    # for i in np.arange(roop):
    #     theta_color[i] = im.make_color(theta[i])

    # im.save_imgs('LLLL_symm_area-9_k-0.01', theta_color)
