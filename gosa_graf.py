import os

import numpy as np


os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', "result", 'twist_test'))

def r_theta_diff(path1, path2):
    r_10 = np.load(path1 + "r.npy")
    theta_10 = np.load(path1 + "theta.npy")
    r_20 = np.load(path2 + "r.npy")
    theta_20 = np.load(path2 + "theta.npy")
    r_dis = np.abs(r_10 - r_20)
    theta_dis = np.abs(theta_10 - theta_20)
    theta_dis[theta_dis > 0.5] = 1 - theta_dis[theta_dis > 0.5]
    return np.sum(r_dis), np.sum(theta_dis)


paths = ["10h-step-", "20h-step-", "40h-step-", "80h-step-", "160h-step-", "320h-step-", "640h-step-", "1280h-step-", "2560h-step-"]
result = np.empty((2, len(paths) - 1))
for i in range(len(paths) - 1):
    result[0, i], result[1, i] = r_theta_diff(paths[i], paths[i + 1])


np.savetxt('out.csv', result, delimiter=',')
