import numpy as np
import math
import time

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
    N = 40
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
    sgm = set_param(math.log(min_sgm), math.log(max_sgm), N)
    sgm = [math.exp(s) for s in sgm]
    prob = calc_gl(N, xi, sgm, data, error, thr)

    pp = np.sum(prob)  # 尤度の合計

    sum = 0  # 累積尤度
    sorted_array = []  # sorted_array = [[probの値, [index1, index2]], ...] ← これが目標
    for _ in range(N * N):
        if _ == 1:
            break
        max_index = np.unravel_index(np.argmax(prob), prob.shape)
        sorted_array.append([prob[max_index[0], max_index[1]], max_index])
        prob[max_index[0], max_index[1]] = 0
    for i in range(N * N):  # N*N回ループを回して, 全てのprob[i, j]に対して累積の尤度？てきなものを計算する
        max_value = sorted_array[i][0] / pp
        # 100再現期待値
        x = xi[sorted_array[i][1][0]]
        s = sgm[sorted_array[i][1][1]]
        # 定数
        a = 100 * 24 * 365 * n0 / n
        rv = thr + s * (a ** x - 1) / x
        if i == 0:
            RV = rv
            break
        sum += max_value

    return RV