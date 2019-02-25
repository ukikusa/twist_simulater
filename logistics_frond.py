# -*-coding: utf-8 -*-
"""logistics growth."""

import os


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def logistic(t, N0, K, r):
    """ロジスティック方程式

    Args:
        t: 計算する配列．
        N0: 初期値
        K: 密度効果
        r: 増殖率
    """
    t = np.array(t)
    N = K / ((1 + K / N0 - 1) * np.exp(-r * t) + 1)
    return N


def logistic_rs(time, N0, K, rs, save):
    if os.path.exists(save) is False:
        os.makedirs(save)
    label = []
    Ns = np.empty((rs.size, time.size), np.uint16)
    for i, r in enumerate(rs):
        label.append('r-' + "{0:.3f}".format(r))
        Ns[i] = logistic(time, N0=N0, K=K, r=r)
    data = pd.DataFrame(data=Ns, index=label, columns=time)
    plt.figure()
    data.T.plot()
    plt.savefig(os.path.join(save, 'logistic_N0-' + str(N0) + '_K-' + str(K) + '.pdf'))
    plt.close()
    data.to_csv(os.path.join(save, 'logistic_N0-' + str(N0) + '_K-' + str(K) + '.csv'))
    return data


def logistic_Ks(time, N0, Ks, r, save):
    if os.path.exists(save) is False:
        os.makedirs(save)
    label = []
    Ns = np.empty((Ks.size, time.size), np.uint16)
    for i, K in enumerate(Ks):
        label.append('K-' + str(K))
        Ns[i] = logistic(time, N0=N0, K=K, r=r)
    data = pd.DataFrame(data=Ns, index=label, columns=time)
    plt.figure()
    data.T.plot()
    plt.savefig(os.path.join(save, 'logistic_N0-' + str(N0) + '_r-' + str(r) + '.pdf'))
    plt.close()
    data.to_csv(os.path.join(save, 'logistic_N0-' + str(N0) + '_r-' + str(r) + '.csv'))
    return data


if __name__ == "__main__":
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python'))
    # カレントディレクトリの変更．

    time = np.arange(0, 168, 1)
    N0 = 50
    K = 4000
    rs = np.arange(0.04, 0.105, 0.005)

    save = os.path.join('result', 'frond_area', 'N0-' + str(N0) + '_K-' + str(K))
    logistic_rs(time, N0, K, rs, save)

    r = 0.06
    Ks = np.arange(2500, 5000, 500, dtype=np.uint16)
    save = os.path.join('result', 'frond_area', 'N0-' + str(N0) + '_r-' + str(r))
    logistic_Ks(time, N0, Ks, r, save)
