"""
各モデルについて, POTを算出する.
現在気候も将来気候も25年分だけ算出する.
"""
import pygrib
import os
import numpy as np
import matplotlib.pyplot as plt
from csv import reader
import cartopy.crs as ccrs
import copy
import matplotlib as mpl
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as ticker


def calc_pot_all(dir_path):

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

    DIR_PATH = dir_path
    DIR = os.listdir(DIR_PATH)

    PARAM = ['100', '101', '102', '103', '104', '105', '106', '107', '108']

    for filename in DIR:
        if filename[:3] != "Jpn":  # 日本のデータのみ使用
            continue
        grbs = pygrib.open(DIR_PATH + filename)
        # 描画(現在気候の最大値)
        fig = plt.figure(figsize=(20, 24))
        levels = np.arange(0, 15, 0.1)
        for i in range(len(PARAM)):
            if i == 7:
                levels = np.arange(0, 360, 1)
                cmap = copy.copy(mpl.cm.get_cmap("jet"))
            else:
                levels = np.arange(0, 20, 0.1)
                cmap = copy.copy(mpl.cm.get_cmap("jet"))
            param = PARAM[i]
            grbs_param = grbs.select(parameterName=param)
            min_data = np.zeros((79, 79))
            # for a in range(79):
            #     for b in range(79):
            #         min_data[a][b] = 100
            for grb in grbs_param:
                data = grb.data()[0].filled(fill_value=0)
                for a in range(79):
                    for b in range(79):
                        min_data[a][b] = max(min_data[a][b], data[a][b])

            ax = fig.add_subplot(3, 3, i + 1, projection=ccrs.PlateCarree())
            a = plt.contourf(lons, lats, min_data, levels=levels, cmap=cmap, extend='max')
            c_bar = plt.colorbar(a)
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
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, alpha=0.5) # 経度線・緯度線ラベルを無効
            gl.xlocator = ticker.FixedLocator([130, 150]) # 経度線
            gl.ylocator = ticker.FixedLocator([15, 30, 45]) # 緯度線
            ax.coastlines()
            ax.set_title(param, fontsize=18)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.savefig('snapshot')
    plt.show()
    return
