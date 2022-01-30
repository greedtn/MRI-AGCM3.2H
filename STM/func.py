import numpy as np
import math


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
    # prob = np.zeros((n, n))
    # for i in range(n):  # ξの添字
    #     for j in range(n):  # σの添字
    #         sum = 0  # 対数尤度
    #         xi_ = xi[i]
    #         sgm_ = sgm[j]
    #         for k in range(len(data)):
    #             data_ = data[k]
    #             error_ = error[k]
    #             y1 = data_ - error_ - thr
    #             y2 = data_ + error_ - thr
    #             # ξが0かどうかでCDFの式が変わる
    #             if xi_ == 0:
    #                 sum += math.log(math.exp(-y2 / sgm_) - math.exp(-y1 / sgm_))
    #             else:
    #                 sum += (math.log(1 + max(0, xi_ * y2 / sgm_)) - math.log(1 + max(0, xi_ * y1 / sgm_))) / xi_
    #         prob[i, j] = -sum
    # return prob

    prob = np.zeros((n, n))
    for i in range(n):  # ξの添字
        for j in range(n):  # σの添字
            sum = 0
            xi_ = xi[i]
            sgm_ = sgm[j]
            ok = True
            for k in range(len(data)):
                y = data[k] - thr
                if 1 + xi_ * y / sgm_ <= 0:  # この条件を満たさないものはNG
                    ok = False
                    break
                sum += math.log(1 + (xi_ * y / sgm_))
            if (ok):
                prob[i, j] = len(data) * math.log(sgm_) + ((1 / xi_) + 1) * sum
            else:
                prob[i, j] = 0
            print('xi:', xi_, 'sgm:', sgm_, '対数尤度:', prob[i][j])
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


def lwm_gpd(data, error, thr, n, n0, con):
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
    print("最大対数尤度:", max_p)

    max_index = np.unravel_index(np.argmax(prob), prob.shape)
    print(max_index, 'xi:', xi[max_index[0]], 'sgm:', sgm[max_index[1]])

    return

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
    N = 200
    # パラメータの範囲を絞って、粒度を細かくした
    xi = set_param(min_xi, max_xi, N)
    sgm = set_param(math.log(min_sgm), math.log(max_sgm), N)
    sgm = [math.exp(s) for s in sgm]
    prob = calc_gl(N, xi, sgm, data, error, thr)

    pp = np.sum(prob)  # 尤度の合計

    sum = 0
    rv_min = 100  # 再現期待値のcon%信頼区間の最小値
    rv_max = 0  # 再現期待値のcon%信頼区間の最大値
    sum_prob = np.zeros((N, N))  # 累積尤度を格納する2d-array
    sorted_array = []  # sorted_array = [[probの値, [index1, index2]], ...] ← これが目標
    for _ in range(N * N):
        max_index = np.unravel_index(np.argmax(prob), prob.shape)
        sorted_array.append([prob[max_index[0], max_index[1]], max_index])
        prob[max_index[0], max_index[1]] = 0
    # 全ての格子点に対して、累積尤度的なものを計算する
    for i in range(N * N):  # N*N回ループを回して, 全てのprob[i, j]に対して累積の尤度？てきなものを計算する
        max_value = sorted_array[i][0] / pp
        # 100再現期待値
        x = xi[sorted_array[i][1][0]]
        s = sgm[sorted_array[i][1][1]]
        rv = thr + s * ((100 * 365 * 24 * 79 * 79 * n0 / n) ** x - 1) / x
        if i == 0:
            RV = rv  # 最尤推定値
            print("最尤推定", "ξ:", x, "σ:", s, "RV:", RV)
        sum += max_value
        if sum < con:
            rv_min = min(rv_min, rv)
            rv_max = max(rv_max, rv)
        else:
            break
        sum_prob[sorted_array[i][1]] = sum

    return [rv_min, RV, rv_max]
