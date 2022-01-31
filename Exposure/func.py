"""
decluster後に15.0mを超えるものが発生したら, その時間のsnapshotを保存する
出力としては, 79*79行で, それぞれの行のサイズはSTMの回数になっていて, 
その中身はSTMが発生したときのその場所の波高になる.
"""
import pygrib
import os
import csv
import numpy as np
import math


def calc_exposure(thr, dir_path, param_name, model_name):
    THR = thr  # 閾値
    POT = [[0] for _ in range(79 * 79)]  # 閾値を超えるデータ(POT[-1]を使用するために初期値を設定)を格納する2d-array
    POT_IDX = [-168]  # 閾値を超えるデータのindex(POT_IDX[-1]を使用するために初期値を設定)を格納する2d-array
    CNT = 0  # indexカウント用の変数
    NOW = 0  # 進捗管理用の変数

    DIR_PATH = dir_path
    DIR = os.listdir(DIR_PATH)

    CNT = 0
    for filename in DIR:
        if filename[:3] != "Jpn":  # 日本のデータのみ使用
            continue
        grbs = pygrib.open(DIR_PATH + filename)
        grbs = grbs.select(parameterName=param_name)
        for grb in grbs:
            CNT += 1
            data = grb.data()[0].filled(fill_value=0)
            pot = np.where(data > THR)
            if len(pot[0]) == 0:  # 閾値超えがなければskip
                continue
            pot = np.where(data == np.max(data))  # ここは1ヶ所しかない前提(複数箇所ある場合は最初)
            m = pot[0][0]
            n = pot[1][0]
            d = data[m][n]  # STMがが起きた場所
            # decluster(1週間以上間隔を空ける)
            if CNT > POT_IDX[-1] + 24 * 7:
                for i in range(79):
                    for j in range(79):
                        POT[79 * i + j].append(math.floor(data[i][j] * 10 ** 2) / (10 ** 2))
                POT_IDX.append(CNT)
            else:
                if d > POT[79 * m + n][-1]:
                    for i in range(79):
                        for j in range(79):
                            POT[79 * i + j][-1] = math.floor(data[i][j] * 10 ** 2) / (10 ** 2)
                    POT_IDX[-1] = CNT

        NOW += 1
        print(f'----- {NOW} / 300 done')

        if NOW >= 300:
            break

    # 初期値を削除
    for i in range(79 * 79):
        POT[i].pop(0)
    POT_IDX.pop(0)

    print("STMの発生回数:", len(POT_IDX))
    
    # 書き出し
    with open('../Ex_csv/' + model_name + '_ex.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(POT)
    with open('../Ex_csv/' + model_name + '_ex_idx.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(POT_IDX)
    return
