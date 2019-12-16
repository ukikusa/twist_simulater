# -*-coding: utf-8 -*-
"""
    Twist chain oscillator simulations. Version using GPU.

It is based on Poincare oscillators.
Same model as proposed by 
{Myung, Jihwan, et al.Nature communications 9.1 (2018): 1062.}
"""

# import sys
import datetime
import os

import chainer.functions
import cupy as cp
import cv2
import numpy as np
import pandas as pd

import image_analysis as im


def oscillator_2D(x, y, kernel, omega, lambda_, epsilon, A=1, mask=False):
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
    p = int(kernel.shape[2] * 0.5)
    field_x = chainer.functions.convolution_2d(
        x.reshape(1, 1, r_shape0, r_shape1), kernel, pad=p
    )
    field_y = chainer.functions.convolution_2d(
        y.reshape(1, 1, r_shape0, r_shape1), kernel, pad=p
    )
    field_x = cp.array(field_x.array, dtype=cp.float64).reshape(r.shape)
    field_y = cp.array(field_y.array, dtype=cp.float64).reshape(r.shape)
    if mask is not False:
        field_x[mask == 0] = 0
        field_y[mask == 0] = 0
    fx = (lambda_ * x - epsilon * y) * (A - r) - omega * y + field_x
    fy = (lambda_ * y + epsilon * x) * (A - r) + omega * x + field_y
    return fx, fy


def RK4(x, y, h, **kwargs):
    """Solve oscillator_2D by Runge–Kutta method.

    Args:
        x: array of x coordinates
        y: array of y coordinates
        h: Step width

    Returns:
        new_x, new_y
    """
    # omega = omega * h
    # kernel = kernel * h
    k1_x, k1_y = oscillator_2D(x, y, **kwargs)
    k2_x, k2_y = oscillator_2D(x + k1_x * 0.5 * h, y + k1_y * 0.5 * h, **kwargs)
    k3_x, k3_y = oscillator_2D(x + k2_x * 0.5 * h, y + k2_y * 0.5 * h, **kwargs)
    k4_x, k4_y = oscillator_2D(x + k3_x * h, y + k3_y * h, **kwargs)
    new_x = x + h / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    new_y = y + h / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

    return new_x, new_y


def oscillator_2D_RK4(x, y, h, step_count, save_step, **kwargs):
    """"Solve oscillator_2D by Runge–Kutta method.

    Args:
        x: array of x coordinates
        y: array of y coordinates
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
        x, y = RK4(x, y, h, **kwargs)
    return xs, ys


def xy2theta_amp_save(
    xs,
    ys,
    save="twin2d",
    kernel=0,
    omega=0,
    lambda_=0,
    epsilon=0,
    A=0,
    h=0,
    step_count=0,
    save_step=0,
):
    """Function to convert Cartesian coordinates to polar coordinates.

    Args:
        xs: X coordinate
        ys: Y coordinate
        save: Save folder path.

    Returns:
        theta, r
    """
    # Calculation theta and r
    theta = cp.arctan2(ys, xs) / cp.pi * 0.5
    theta[theta < 0] = 1 + theta[theta < 0]
    xy = cp.stack([xs, ys])
    r = cp.linalg.norm(xy, axis=0)
    if save is not False:  # save something
        dirctroy = os.path.join(save, datetime.datetime.today().strftime("%y%m%d"))
        if not os.path.exists(dirctroy):
            os.makedirs(dirctroy)
        cp.save(os.path.join(dirctroy, "theta.npy"), theta)
        cp.save(os.path.join(dirctroy, "r.npy"), r)
        # 位相を色相に変換する．振幅が0のときは黒(明度0)にする．
        theta[r == 0] = -1
        color = im.make_colors(cp.asnumpy(theta), black=-1)
        im.save_imgs(
            save_folder=dirctroy, img=color, file_name="color_theta", stack=True
        )
        # save r by color
        r01 = cp.copy(r)
        print(r01.dtype)
        r_max, r_min = cp.max(r), cp.min(r[r != 0])
        r01 = (r01 - r_min) / (r_max - r_min) * 0.7
        r01[r == 0] = -1
        color = im.make_colors(cp.asnumpy(r01), black=-1)
        im.save_imgs(
            save_folder=dirctroy,
            img=color,
            file_name="color_r" + str(r_max) + "-" + str(r_min),
            stack=True,
        )
        # save parameter
        parameter = {
            "lambda": lambda_,
            "epsilon": epsilon,
            "A": A,
            "h": h,
            "step_count": step_count,
            "save_step": save_step,
        }
        pd.DataFrame(parameter, index=["value"]).to_csv(
            os.path.join(dirctroy, "parameter.csv")
        )
        kernel = np.savetxt(
            os.path.join(dirctroy, "kernel.csv"),
            cp.asnumpy(kernel.reshape(kernel.shape[2:4])),
            delimiter=",",
        )
        np.savetxt(
            os.path.join(dirctroy, "omega.csv"), cp.asnumpy(omega), delimiter=","
        )
    return theta, r


    def gaussian_kernel(ksize, sigma=False):  # ガウシアン行列作る．
    if sigma is False:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        print('SIGMA = ' + str(sigma))
    kernel_1d = np.empty((ksize, 1), dtype=np.float64)  # 1次元で作って拡張
    for i in np.arange(ksize):
        kernel_1d[i] = np.exp(-1 * np.power(i - (ksize - 1) * 0.5, 2) / (2 * np.power(sigma, 2)))
    kernel_2d = kernel_1d.dot(kernel_1d.T)
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d


if __name__ == "__main__":
    mask = os.path.join(
        "/hdd1",
        "Users",
        "kenya",
        "Labo",
        "keisan",
        "python",
        "result",
        "twist_2d",
        "frond_form_85.png",
    )
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    # R = 250  # 解析のサイズ
    # C = 150  # 解析のサイズ
    R, C = mask.shape
    # 細胞サイズを20μ，フロンドサイズを3mm * 5mm とした時．
    A = 1  # amplitude
    lambda_ = 0.02
    epsilon = -0.005  # twist h
    omega = 2 * cp.pi / 25.5 + 0.01 * cp.random.randn(R, C)  # 角速度
    theta = cp.random.rand(R, C) * 0  # 初期位相
    x = cp.cos(theta).astype(cp.float64) * 15
    y = cp.sin(theta).astype(cp.float64) * 15
    kernel = cp.ones((3, 3), dtype=cp.float64) * 0.05
    kernel[1, 1] = 0
    kernel = kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1])
    h = 0.5 ** 5  # 時間刻み huer
    save_step = int(1 / h)  # 保存頻度
    step_count = int(10 * 24 / h)  # ループ回数

    if mask is not False:
        x[mask == 0] = 0
        y[mask == 0] = 0
    for i in range(1):
        save = os.path.join(
            "/hdd1",
            "Users",
            "kenya",
            "Labo",
            "keisan",
            "python",
            "result",
            "twist_2d",
            "omega-rand",
        )
        xs, ys = oscillator_2D_RK4(
            x,
            y,
            kernel=kernel,
            omega=omega,
            lambda_=lambda_,
            epsilon=epsilon,
            A=A,
            h=h,
            step_count=step_count,
            save_step=save_step,
            mask=mask,
        )
        xy2theta_amp_save(
            xs,
            ys,
            save=save,
            kernel=kernel,
            omega=omega,
            lambda_=lambda_,
            epsilon=epsilon,
            A=A,
            h=h,
            step_count=step_count,
            save_step=save_step,
        )

