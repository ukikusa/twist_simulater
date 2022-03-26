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
    r = np.linalg.norm(np.array([x, y]), axis=0)
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
    x_shape0, x_shape1 = x.shape
    save_idx = range(0, step_count, save_step)
    save = 0
    xs = np.empty((len(save_idx), x_shape0, x_shape1), dtype=np.float64)
    ys = np.empty_like(xs)
    for i in range(step_count):
        if save_idx[save] == i:
            xs[save], ys[save] = x, y
            save = save + 1
            if save >= len(save_idx):
                break

        x, y = RK4(x, y, kernel, omega, A, h)
    return xs, ys


def xy2theta_amp(xs, ys, save=True):
    theta = np.arctan(ys / xs) / np.pi * 0.5
    theta[theta < 0] = 1 + theta[theta < 0]
    r = np.linalg.norm(np.array([xs, ys]), axis=0)
    if save is not False:
        np.save(save + "-theta.npy", theta)
        np.save(save + "-r.npy", r)
    return theta, r

if __name__ == '__main__':
    step_count = 240  # ループ回数
    h = 1  # 時間刻み huer
    R = 250  # 解析のサイズ
    C = 150  # 解析のサイズ
    # 細胞サイズを20μ，フロンドサイズを3mm * 5mm とした時．
    A = 1  # amplitude
    omega = 2 * np.pi / 24  # + 0.01 * np.random.randn(R, C)  # 角速度
    theta = np.random.rand(R, C) * 0  # 初期位相
    x = np.cos(theta).astype(np.float64) * 1
    y = np.sin(theta).astype(np.float64) * 1
    kernel = np.ones((3, 3)) * 10 ** (-9)
    kernel[1, 1] = 0
    save_step = 1
    for i in range(10):
        print(i)
        h = 0.1 * 0.5 ** i
        omega = 2 * np.pi / 24 * h
        save_step = int(1 / h)
        step_count = int(30 / h)
        save = "/hdd1/Users/kenya/Labo/keisan/python/result/twist_test/" + str(int(1 / h)) + "h-step"
        xs, ys = oscillator_2D_RK4(x, y, kernel, omega, A, h, step_count, save_step)
        xy2theta_amp(xs, ys, save=save)

    # theta_color = np.empty((theta.shape[0], theta.shape[1], theta.shape[2], 3), dtype=np.int16)
    # for i in np.arange(roop):
    #     theta_color[i] = im.make_color(theta[i])

    # im.save_imgs('LLLL_symm_area-9_k-0.01', theta_color)
