import pygrib
import os
import csv
from csv import reader
import matplotlib.pyplot as plt
import numpy as np
import math
import time


def calc_pot(thr, dir_path, param_name, output_csv, index):
    """
    POTを算出してCSVに出力する関数
    """
    THR = thr  # 閾値
    POT = [0]  # 閾値を超えるデータ(POT[-1]を使用するために初期値を設定)
    POT_IDX = [-168]  # 閾値を超えるデータのindex(POT_IDX[-1]を使用するために初期値を設定)
    CNT = 0  # indexカウント用の変数

    DIR_PATH = dir_path
    DIR = os.listdir(DIR_PATH)

    for filename in DIR:
        if filename[:3] != "Jpn":
            continue
        print(f'{filename}...now')
        grbs = pygrib.open(DIR_PATH + filename)
        grbs = grbs.select(parameterName=param_name)
        for grb in grbs:
            CNT += 1
            # まずは1地点でやるために, (20, 20)で試す
            # maskされた部分は0で埋める
            m = index // 79
            n = index % 79
            data = grb.data()[0].filled(fill_value=0)[m][n]
            # decluster
            if data > THR:
                if CNT > POT_IDX[-1] + 168:
                    POT.append(data)
                    POT_IDX.append(CNT)
                else:
                    if data > POT[-1]:
                        POT[-1] = data
                        POT_IDX[-1] = CNT

    # 初期値を削除
    POT.pop(0)
    POT_IDX.pop(0)

    # 書き出し
    f = open(output_csv, 'w')
    writer = csv.writer(f)
    writer.writerow(POT)
    writer.writerow(POT_IDX)
    writer.writerow([CNT])
    f.close()

    return POT, POT_IDX, CNT, len(POT)


def plot(model_name, img_title, index):

    with open('../csv2/' + model_name + '_CNT.csv', 'r') as csv_file:
        csv_reader = reader(csv_file)
        CNT = int(list(csv_reader)[0][0])

    with open('../csv2/' + model_name + '_POT.csv', 'r') as csv_file:
        csv_reader = reader(csv_file)
        POT = list(csv_reader)[index]

    with open('../csv2/' + model_name + '_POT_IDX.csv', 'r') as csv_file:
        csv_reader = reader(csv_file)
        POT_IDX = list(csv_reader)[index]

    # データ数を削減する
    s = []
    for i in range(len(POT)):
        s.append([float(POT[i]), int(POT_IDX[i])])
    s = sorted(s, reverse=True)
    x = []
    y = []
    if model_name[:3] == 'HPA':
        period = 31
    else:
        period = 25
    p = min(period * 2, len(POT))
    for i in range(p):
        x.append(int(s[i][1]))
        y.append(float(s[i][0]))

    print("プロット数:", len(x))
    plt.plot(x, y, 'o')
    plt.xticks([])
    # plt.yticks([])
    plt.xlabel("time")
    plt.ylabel("Hs[m]")
    plt.title(img_title)
    plt.show()

    return y, CNT


def calc_gl(n, xi, sgm, data, error, thr):
    """
    各格子点周りの尤度を計算する（まだ対数尤度にはできていない)

    Args:
        n (int): 格子点の数の平方根(n*n個の格子点を作るので)
        xi (list): ξ
        sgm (list): σ
        data (list): 閾値を超えたデータ
        error (list): 誤差(要素数は1 or n)
        thr (int): 閾値

    Returns:
        prob (2darray): prob[i, j]は格子点(i, j)周りの尤度

    """
    prob = np.zeros((n, n))
    for i in range(n):  # ξの添字
        for j in range(n):  # σの添字
            cdf = 10 ** len(data)  # 最初に大きい値にしておく
            xi_ = xi[i]
            sgm_ = sgm[j]
            for k in range(len(data)):
                data_ = data[k]
                error_ = error[k]
                # ξが0かどうかでCDFの式が変わる
                if xi_ == 0:
                    cdf1 = 1 - math.exp(- (max(0, data_ - error_ - thr)) / sgm_)
                    cdf2 = 1 - math.exp(- (max(0, data_ + error_ - thr)) / sgm_)
                else:
                    cdf1 = 1 - max(0, 1 + xi_ * max(0, data_ - error_ - thr) / sgm_) ** (-1 / xi_)
                    cdf2 = 1 - max(0, 1 + xi_ * max(0, data_ + error_ - thr) / sgm_) ** (-1 / xi_)
                cdf = cdf * (cdf2 - cdf1)
            prob[i, j] = cdf
    return prob


def set_param(min_par, max_par, n):
    """
    パラメータの設定（最小値から最大値までをn分割する)

    Args:
        min_par (int): 最小値
        max_par (int): 最大値(この値は含まない)
        n (int): 分割数

    Returns:
        params (list): 設定されたパラメータ

    """
    return np.linspace(min_par, max_par, n)


def lwm_gpd(data, error, thr, period, RP, n, n0, con, img_name):
    """
    PPD(POSTERIOR PREDICTIVE DISTRIBUTION)の描画と、信頼区間ごとの尤度を描画する

    Args:
        data (list): データ
        error (list): 誤差
        thr (int): 閾値
        period (int): 期間
        RP (list): 再現期間(要素数は1 or n)
        n (int) : データの総数
        n0 (int) : 閾値を超えるデータの数
        con (float) : 信頼区間(0.9→90%)

    Returns:
        描画する(x, ppd)
        描画する(ξ, logσ)
        Fval(list): 再現期待値(RPの要素数個分だけ出てくる)
    """
    start = time.time()

    # 誤差は1つだけしか与えられなくても、1*nの配列に変換する
    if len(error) == 1:
        for i in range(len(data) - 1):
            error.append(error[0])

    # 格子点の粒度
    N = 100
    # ξとσをセット
    xi = set_param(-5, 5, N)
    sgm = set_param(math.log(0.01), math.log(10), N)
    sgm = [math.exp(s) for s in sgm]
    prob = calc_gl(N, xi, sgm, data, error, thr)

    # 最大尤度
    max_p = np.max(prob)
    # 最小尤度(これ以下の値は除外する→ξとσの範囲を絞るため)
    min_p = max_p / 10 ** 8
    # min_pよりも大きい値を取るindexのリスト
    xi_sub = np.where(prob > min_p)[0]
    xi_sub = sorted(list(set(xi_sub)))  # 重複削除
    max_xi = xi[xi_sub[-1]]
    min_xi = xi[xi_sub[0]]
    if min_xi == max_xi:
        min_xi -= 0.3
        max_xi += 0.3
    # min_pよりも大きい値を取るindexのリスト
    sgm_sub = np.where(prob > min_p)[1]
    sgm_sub = sorted(list(set(sgm_sub)))  # 重複削除
    max_sgm = sgm[sgm_sub[-1]]
    min_sgm = sgm[sgm_sub[0]]
    if min_sgm == max_sgm:
        min_sgm = min_sgm / 3
        max_sgm = max_sgm * 3

    # 粒度
    N = 100
    # パラメータの範囲を絞って、粒度を細かくした
    xi = set_param(min_xi, max_xi, N)
    # print("xi:", min(xi), "~", max(xi))
    sgm = set_param(math.log(min_sgm), math.log(max_sgm), N)
    sgm = [math.exp(s) for s in sgm]
    prob = calc_gl(N, xi, sgm, data, error, thr)
    # print("最大尤度を取るインデックス: ", np.unravel_index(np.argmax(prob), prob.shape))

    pp = np.sum(prob)  # 尤度の合計

    sum = 0  # 累積尤度
    rv_min = 100  # 再現期待値の95%信頼区間の最小値
    rv_max = 0  # 再現期待値の95%信頼区間の最大値
    sum_prob = np.zeros((N, N))  # 累積尤度を格納する2d-array
    sorted_array = []  # sorted_array = [[probの値, [index1, index2]], ...] ← これが目標
    for _ in range(N * N):
        if _ == 1:
            break
        max_index = np.unravel_index(np.argmax(prob), prob.shape)
        sorted_array.append([prob[max_index[0], max_index[1]], max_index])
        prob[max_index[0], max_index[1]] = 0
    # 全ての格子点に対して、累積尤度的なものを計算する
    # print("n0, n, n0 / n", n0, n, n0 / n)
    for i in range(N * N):  # N*N回ループを回して, 全てのprob[i, j]に対して累積の尤度？てきなものを計算する
        max_value = sorted_array[i][0] / pp
        # 100再現期待値
        x = xi[sorted_array[i][1][0]]
        s = sgm[sorted_array[i][1][1]]
        # 定数
        a = 100 * 24 * 365 * n0 / n
        rv = thr + s * (a ** x - 1) / x
        if i == 0:
            RV = rv  # 最尤推定値
            # print("最尤推定", "ξ:", x, "σ:", s, "RV:", RV)
            break
        sum += max_value
        if sum < con:
            if (rv > 30):
                print("sgm: ", s, "xi: ", x, "RV: ", rv, "sum: ", sum)
            rv_min = min(rv_min, rv)
            rv_max = max(rv_max, rv)
        # else:
        #     break
        sum_prob[sorted_array[i][1]] = sum
    # print("sum:", sum)

    # # 等高線の描画
    # log_sgm = np.array([math.log(s) for s in sgm])
    # X, Y = np.meshgrid(xi, log_sgm)
    # Z = np.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    #         if sum_prob[i, j] > 0:
    #             Z[j][i] = sum_prob[i, j]  # 上下左右が逆になるのでiとjを入れ替える
    #         else:
    #             Z[j][i] = 1
    # plt.figure(figsize=(16, 8))
    # plt.title("Thr = 9.0")
    # plt.xlabel("ξ")
    # plt.ylabel("logσ")
    # # plt.xlim([-0.5, 0.5])
    # # plt.ylim([-0.8, 0.8])
    # cont = plt.contour(
    #     X, Y, Z, levels=[0.1, 0.3, 0.5, 0.7, 0.95], colors='black')
    # cont.clabel(fmt='%1.2f', fontsize=10)
    # plt.gca().set_aspect('equal')
    # plt.show()
    # plt.savefig(img_name)
    # plt.show()

    return RV
