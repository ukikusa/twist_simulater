# -*-coding: utf-8 -*-
"""
    Twist chain oscillator simulations. Version using GPU.

It is based on Poincare oscillators.
Same model as proposed by {Myung, Jihwan, et al. "The choroid plexus is an important circadian clock component." Nature communications 9.1 (2018): 1062.}
"""

# import sys
import os

import chainer.functions
import cupy as cp
import image_analysis as im


def oscillator_2D(x, y, kernel, omega, lambda_, epsilon, A=1):
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
    r = cp.linalg.norm(cp.stack([x, y]), axis=0)
    r_shape0, r_shape1 = r.shape
    field_x = chainer.functions.convolution_2d(x.reshape(1, 1, r_shape0, r_shape1), kernel, pad=1)
    field_y = chainer.functions.convolution_2d(y.reshape(1, 1, r_shape0, r_shape1), kernel, pad=1)
    field_x = cp.array(field_x.array, dtype=cp.float64).reshape(r.shape)
    field_y = cp.array(field_y.array, dtype=cp.float64).reshape(r.shape)
    fx = (lambda_ * x - epsilon * y) * (A - r) - omega * y + field_x
    fy = (lambda_ * y + epsilon * x) * (A - r) + omega * x + field_y
    return fx, fy


def RK4(x, y, kernel, omega, lambda_, epsilon, A, h):
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
    # omega = omega * h
    # kernel = kernel * h
    k1_x, k1_y = oscillator_2D(x, y, kernel, omega, lambda_, epsilon, A)
    k2_x, k2_y = oscillator_2D(x + k1_x * 0.5 * h, y + k1_y * 0.5 * h, kernel, omega, lambda_, epsilon, A)
    k3_x, k3_y = oscillator_2D(x + k2_x * 0.5 * h, y + k2_y * 0.5 * h, kernel, omega, lambda_, epsilon, A)
    k4_x, k4_y = oscillator_2D(x + k3_x * h, y + k3_y * h, kernel, omega, lambda_, epsilon, A)
    new_x, new_y = x + h / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x), y + h / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
    return new_x, new_y


def oscillator_2D_RK4(x, y, kernel, omega, lambda_, epsilon, A, h, step_count, save_step):
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
    xs = cp.empty((len(save_idx), x_shape0, x_shape1), dtype=cp.float64)
    ys = cp.empty_like(xs)
    for i in range(step_count):
        if save_idx[save] == i:
            xs[save], ys[save] = x, y
            save = save + 1
            if save >= len(save_idx):
                break
        x, y = RK4(x, y, kernel, omega, lambda_, epsilon, A, h)
    return xs, ys


def xy2theta_amp(xs, ys, save="twin2d"):
    """Function to convert Cartesian coordinates to polar coordinates.

    Args:
        xs: X coordinate
        ys: Y coordinate
        save: Save folder path.

    Returns:
        theta, r
    """
    theta = cp.arctan2(ys, xs) / cp.pi * 0.5
    theta[theta < 0] = 1 + theta[theta < 0]
    xy = cp.stack([xs, ys])
    r = cp.linalg.norm(xy, axis=0)
    if save is not False:
        dirctroy = os.path.dirname(save)
        if not os.path.exists(dirctroy):
            os.makedirs(dirctroy)
        cp.save(save + "-theta.npy", theta)
        cp.save(save + "-r.npy", r)
        color = im.make_colors(cp.asnumpy(theta))
        if not os.path.exists(save + 'color_theta'):
            os.makedirs(save + 'color_theta')
        im.save_imgs(save + 'color_theta', color)
    return theta, r

if __name__ == '__main__':
    step_count = 240  # ループ回数
    h = 1  # 時間刻み huer
    R = 250  # 解析のサイズ
    C = 150  # 解析のサイズ
    # 細胞サイズを20μ，フロンドサイズを3mm * 5mm とした時．
    A = 1  # amplitude
    lambda_ = 0.02
    epsilon = 0.01  # twist h
    omega = 2 * cp.pi / 24  # + 0.01 * np.random.randn(R, C)  # 角速度
    theta = cp.random.rand(R, C) * 0  # 初期位相
    x = cp.cos(theta).astype(cp.float64) * 1
    y = cp.sin(theta).astype(cp.float64) * 1
    kernel = cp.ones((3, 3), dtype=cp.float64) * 0.1
    kernel[1, 1] = 0
    kernel = kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1])
    save_step = 1
    for i in range(5):
        print(i)
        h = 0.5 ** i
        save_step = int(1 / h)
        step_count = int(30 * 24 / h)
        save = "/hdd1/Users/kenya/Labo/keisan/python/result/twist_test/30day_1hsave_-k01_gpu_lambda002_epsilon001/" + str(int(1 / h)) + "h-step"
        xs, ys = oscillator_2D_RK4(x, y, kernel, omega, lambda_, epsilon, A, h, step_count, save_step)
        xy2theta_amp(xs, ys, save=save)
