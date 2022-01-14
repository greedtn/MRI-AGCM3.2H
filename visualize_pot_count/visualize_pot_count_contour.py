import cartopy.crs as ccrs
import csv
import numpy as np
import pygrib
import matplotlib.pyplot as plt


def func(csv_path, output_name):

    # 緯度経度の設定
    grbs = pygrib.open('/Volumes/HDCL-UT/MRI-AGCM3.2H_WW3_wave/HPA_YS/Jpn_30min.1979010100.grib')
    lats, lons = grbs[1].latlons()
    pot_count = np.zeros((79, 79))

    with open(csv_path) as fp:
        l = list(csv.reader(fp))

    # POTの読み込み
    for i in range(len(l)):
        pot_count[i // 79][i % 79] = len(l[i])

    lat_s = 11
    lat_n = 50
    lon_w = 121
    lon_e = 160

    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.PlateCarree()  # 正距円筒図法を指定
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax = plt.axes(projection=proj)
    # ax.gridlines()
    ax.coastlines(resolution='10m')
    ax.set_extent((lon_w, lon_e, lat_s, lat_n), proj)  # 緯度経度の範囲を指定

    levels = np.arange(0, 1000, 60)  # 等値線の間隔を指定

    CS = ax.contour(lons, lats, pot_count, levels, transform=proj)
    ax.clabel(CS, fmt='%.0f')  # 等値線のラベルを付ける
    ax.set_title("POT COUNT CONTOUR " + output_name)
    plt.savefig("png/POT COUNT COUNTOUR " + output_name)
    plt.show()

    return
