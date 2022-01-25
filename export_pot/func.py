"""
各モデルについて, POTを算出する.
現在気候も将来気候も25年分だけ算出する.
"""
import pygrib
import os
import csv
import numpy as np


def calc_pot_all(thr, dir_path, param_name, model_name):
    THR = thr  # 閾値
    POT = [[0] for _ in range(79 * 79)]  # 閾値を超えるデータ(POT[-1]を使用するために初期値を設定)を格納する2d-array
    POT_IDX = [[-168] for _ in range(79 * 79)]  # 閾値を超えるデータのindex(POT_IDX[-1]を使用するために初期値を設定)を格納する2d-array
    CNT = 0  # indexカウント用の変数
    NOW = 0  # 進捗管理用の変数

    DIR_PATH = dir_path
    DIR = os.listdir(DIR_PATH)

    CNT = 0
    for filename in DIR:
        CNT += 1
        if CNT >= 5:
            break
        if filename[:3] != "Jpn":  # 日本のデータのみ使用
            continue
        grbs = pygrib.open(DIR_PATH + filename)
        grbs = grbs.select(parameterName=param_name)
        cnt = 0
        for grb in grbs:
            cnt += 1
            if cnt >= 10:
                break
            print(grb.data()[0])
            CNT += 1
            data = grb.data()[0].filled(fill_value=0)
            pot = np.where(data > THR)
            for i in range(len(pot[0])):
                m = pot[0][i]
                n = pot[1][i]
                d = data[m][n]
                # decluster(1週間以上間隔を空ける)
                if CNT > POT_IDX[79 * m + n][-1] + 24 * 7:
                    POT[79 * m + n].append(d)
                    POT_IDX[79 * m + n].append(CNT)
                else:
                    if d > POT[79 * m + n][-1]:
                        POT[79 * m + n][-1] = d
                        POT_IDX[79 * m + n][-1] = CNT

        NOW += 1
        print(f'----- {NOW} / 300 done')

        if NOW >= 300:
            break

    # 初期値を削除
    for i in range(79 * 79):
        POT[i].pop(0)
        POT_IDX[i].pop(0)

    # 書き出し
    with open('../pot_csv/' + model_name + '_POT_DATA.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(POT)
    with open('../pot_csv/' + model_name + '_POT_INDEX.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(POT_IDX)
    print(CNT)
    return
