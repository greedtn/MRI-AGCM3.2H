"""
<calc_exposure>
decluster後に15.0mを超えるものを発見したら, その時間のsnapshotを保存する.
出力としては, 79*79行で, それぞれの行のサイズはSTMの回数になっていて, 
その中身はSTMが発生したときのその場所の波高になる.また, STM発生時のCNTも出力する.

<calc_max>
上のCNTの前後1weekを探索して, 前後1weekの各点の最大値を探索する.
それがExposureの定義である.
"""
import pygrib
import os
import csv
import numpy as np
import math
from csv import reader
import cartopy.crs as ccrs
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as ticker
import PIL


def calc_exposure(thr, dir_path, param_name, model_name):
    """ STM 発生時のsnapshot """

    THR = thr  # 閾値
    POT = [[0] for _ in range(79 * 79)]  # STM発生時のsnapshotを格納する2d-array
    POT_IDX = [-168]  # STM発生時のCNTを格納する1d-array
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
            pot = np.where(data > THR)
            if len(pot[0]) == 0:  # 閾値超えがなければskip
                continue
            pot = np.where(data == np.max(data))  # ここは1ヶ所しかない前提(複数箇所ある場合は最初)
            m = pot[0][0]
            n = pot[1][0]
            d = data[m][n]  # STMがが起きた場所
            # decluster(1週間以上間隔を空ける)
            if CNT > POT_IDX[-1] + 24 * 7:  # 新規
                for i in range(79):
                    for j in range(79):
                        POT[79 * i + j].append(math.floor(data[i][j] * 10 ** 2) / (10 ** 2))
                POT_IDX.append(CNT)
            else:  # 編集
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


def calc_max(dir_path, model_name, param_name):
    """ CNTより前後1weekの各点での最大値を探索 """

    MAX_DATA = [[0] for _ in range(79 * 79)]  # STM発生前後1wの各地点での最大値を格納する
    with open('../Ex_csv/' + model_name + '_ex_idx.csv', 'r') as csv_file:
        csv_reader = reader(csv_file)
        Ex_IDX = list(csv_reader)[0]
    CNT = 0  # 通し番号
    NOW = 0  # 完了済みファイル数のカウント

    DIR_PATH = dir_path
    DIR = os.listdir(DIR_PATH)
    Ex = 0  # これを1つずつずらしていく
    is_STM = False
    is_DONE = False

    for filename in DIR:
        if filename[:3] != "Jpn":  # 日本のデータのみ使用
            continue
        grbs = pygrib.open(DIR_PATH + filename)
        grbs = grbs.select(parameterName=param_name)
        for grb in grbs:
            CNT += 1
            if CNT == -24 * 7 + int(Ex_IDX[Ex]):  # STMの前後1wに突入
                is_STM = True
                TEMP_MAX = np.zeros((79, 79))  # 初期化
            if is_STM:
                data = grb.data()[0].filled(fill_value=0)
                TEMP_MAX = np.maximum(TEMP_MAX, data)
            if CNT == int(Ex_IDX[Ex]) + 24 * 7:  # STMの前後1w終了
                is_STM = False
                for i in range(79 * 79):
                    MAX_DATA[i].append(math.floor(TEMP_MAX[i // 79][i % 79] * 10 ** 2) / (10 ** 2))
                Ex += 1  # 次のイベントに移動
                if Ex >= len(Ex_IDX):
                    is_DONE = True
                    break
        if is_DONE:
            break
        NOW += 1
        print(f'----- {NOW} / 300 done')
        if NOW >= 300:
            break

    if is_STM:  # 最後にSTMの前後1w終了時の処理がまだなされていなかった場合
        is_STM = False
        for i in range(79 * 79):
            MAX_DATA[i].append(math.floor(TEMP_MAX[i // 79][i % 79] * 10 ** 2) / (10 ** 2))
    for i in range(79 * 79):
        MAX_DATA[i].pop(0)  # 先頭の0削除
    with open('../Ex_csv/' + model_name + '_max.csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(MAX_DATA)
    return


def ex_visualize(model):
    """ GIF作成 """

    with open('../lats.csv', 'r') as csv_file:
        csv_reader = reader(csv_file)
        lats = list(csv_reader)
    with open('../lons.csv', 'r') as csv_file:
        csv_reader = reader(csv_file)
        lons = list(csv_reader)

    for i in range(79):
        for j in range(79):
            lats[i][j] = float(lats[i][j])
            lons[i][j] = float(lons[i][j])

    with open('../Ex_csv/' + model + '_ex_ratio.csv', 'r') as csv_file:
        csv_reader = reader(csv_file)
        STM = list(csv_reader)
    print(len(STM[0]))
    for i in range(len(STM[0])):
        stm = np.zeros((79, 79))
        for j in range(79):
            for k in range(79):
                if float(STM[j * 79 + k][i]) > 0:
                    stm[j][k] = float(STM[j * 79 + k][i])
                else:
                    stm[j][k] = -1
        fig = plt.figure()
        levels = np.arange(0, 1.001, 0.001)
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        a = plt.contourf(lons, lats, stm, levels=levels, cmap=cmap, extend='max')
        c_bar = plt.colorbar(a, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        c_bar.ax.tick_params(labelsize=18)
        c_bar.ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xticks([130, 150], crs=ccrs.PlateCarree())  # gridを引く経度を指定 360にすると0Wが出ない
        ax.set_yticks([15, 30, 45], crs=ccrs.PlateCarree())  # gridを引く緯度を指定
        lon_formatter = LongitudeFormatter(zero_direction_label=True)  # 経度
        lat_formatter = LatitudeFormatter()  # 緯度。formatを指定することも可能
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.axes.tick_params(labelsize=18)
        ax.grid()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, alpha=0.5)  # 経度線・緯度線ラベルを無効
        gl.xlocator = ticker.FixedLocator([130, 150])  # 経度線
        gl.ylocator = ticker.FixedLocator([15, 30, 45])  # 緯度線
        ax.coastlines()
        ax.set_title(str(i) + '/' + str(len(STM[0])), fontsize=18)
        plt.savefig('temp_img/' + str(i) + '.png')
        plt.close()

    # GIF作成
    image_frames = []  # creating a empty list to be appended later on
    days = np.arange(len(STM[0]))
    for k in days:
        new_fram = PIL.Image.open('temp_img/' + str(k) + '.png')
        image_frames.append(new_fram)
    image_frames[0].save(model + '.gif', format='GIF', append_images=image_frames[1:], save_all=True, duration=100, loop=0)

    return
