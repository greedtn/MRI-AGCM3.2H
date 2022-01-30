"""
各モデルについて, 全ての地点の平均値・最大値を算出する.
現在気候も将来気候も25年分だけ算出する.
"""
from tkinter.tix import MAX
import pygrib
import os
import csv
import numpy as np
import math


def analysis(dir_path, param_name, model_name):

    MAX_DATA = np.zeros((79, 79))  # 各点での最大値を格納する
    AVE_DATA = np.zeros((79, 79))  # 各点での25年間の平均値を格納する
    CNT = 0  # indexカウント用の変数
    NOW = 0  # 進捗管理用の変数
    DIR_PATH = dir_path
    DIR = os.listdir(DIR_PATH)

    for filename in DIR:
        if filename[:3] != "Jpn":  # 日本のデータのみ使用
            continue
        grbs = pygrib.open(DIR_PATH + filename)
        grbs = grbs.select(parameterName=param_name)
        for grb in grbs:
            CNT += 1
            data = grb.data()[0].filled(fill_value=0)
            AVE_DATA += data
            MAX_DATA = np.maximum(MAX_DATA, data)

        NOW += 1
        print(f'----- {NOW} / 300 done')

        if NOW >= 300:
            break
    
    AVE_DATA /= CNT  # 平均値に変換

    # 小数第二位までにする
    for i in range(79):
        for j in range(79):
            AVE_DATA[i][j] = math.floor(AVE_DATA[i][j] * 10 ** 2) / (10 ** 2)
            MAX_DATA[i][j] = math.floor(MAX_DATA[i][j] * 10 ** 2) / (10 ** 2)

    # 書き出し
    with open('../ana_csv/' + model_name + '_MAX.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(MAX_DATA)
    with open('../ana_csv/' + model_name + '_AVE.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(AVE_DATA)
    return
