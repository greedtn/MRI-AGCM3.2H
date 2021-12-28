import func
import matplotlib.pyplot as plt
import numpy as np

def compare(model_name):
    # 100年RVを比較する図を描画する

    POT, CNT = func.plot(filename="../csv/HPA_{}.csv".format(model_name), img_title="HPA_{}".format(model_name), output_name="../png/pot/HPA_{}.png".format(model_name))
    POT_LEN = len(POT)
    RV_ = func.lwm_gpd(data=POT, error=[0.005], thr=9.0, period=31, RP=100, n=CNT, n0=POT_LEN, con=0.95, img_name="../png/param/HPA_{}.png".format(model_name))

    POT, CNT = func.plot(filename="../csv/HFA_{}_c0.csv".format(model_name), img_title="HFA_{}_c0".format(model_name), output_name="../png/pot/HFA_{}_c0.png".format(model_name))
    POT_LEN = len(POT)
    RV_0 = func.lwm_gpd(data=POT, error=[0.005], thr=9.0, period=31, RP=100, n=CNT, n0=POT_LEN, con=0.95, img_name="../png/param/HFA_{}_c0.png".format(model_name))

    POT, CNT = func.plot(filename="../csv/HFA_{}_c1.csv".format(model_name), img_title="HFA_{}_c1".format(model_name), output_name="../png/pot/HFA_{}_c1.png".format(model_name))
    POT_LEN = len(POT)
    RV_1 = func.lwm_gpd(data=POT, error=[0.005], thr=9.0, period=31, RP=100, n=CNT, n0=POT_LEN, con=0.95, img_name="../png/param/HFA_{}_c1.png".format(model_name))

    POT, CNT = func.plot(filename="../csv/HFA_{}_c2.csv".format(model_name), img_title="HFA_{}_c2".format(model_name), output_name="../png/pot/HFA_{}_c2.png".format(model_name))
    POT_LEN = len(POT)
    RV_2 = func.lwm_gpd(data=POT, error=[0.005], thr=9.0, period=31, RP=100, n=CNT, n0=POT_LEN, con=0.95, img_name="../png/param/HFA_{}_c2.png".format(model_name))

    POT, CNT = func.plot(filename="../csv/HFA_{}_c3.csv".format(model_name), img_title="HFA_{}_c3".format(model_name), output_name="../png/pot/HFA_{}_c3.png".format(model_name))
    POT_LEN = len(POT)
    RV_3 = func.lwm_gpd(data=POT, error=[0.005], thr=9.0, period=31, RP=100, n=CNT, n0=POT_LEN, con=0.95, img_name="../png/param/HFA_{}_c3.png".format(model_name))

    # y軸方向にのみerrorbarを表示
    plt.figure(figsize=(10,7))
    plt.errorbar(
        x=[1, 2, 3, 4, 5], 
        y=[RV_[1], RV_0[1], RV_1[1], RV_2[1], RV_3[1]], 
        yerr = np.array(
            [
                [RV_[1] - RV_[0], RV_0[1] - RV_0[0], RV_1[1] - RV_1[0], RV_2[1] - RV_2[0], RV_3[1] - RV_3[0]], 
                [RV_[2] - RV_[1], RV_0[2] - RV_0[1], RV_1[2] - RV_1[1], RV_2[2] - RV_2[1], RV_3[2] - RV_3[1]]
            ]
        ), 
        capsize=5, 
        fmt='o', 
        markersize=10, 
        ecolor='black', 
        markeredgecolor="black", 
        color='w'
    )
    plt.ylabel('Hs[m]')
    plt.title('100 year RV of 95% confidence interval {}'.format(model_name))
    plt.xticks([1, 2, 3, 4, 5], ['past', 'c0', 'c1', 'c2', 'c3'])
    plt.savefig('../png/error_bar/{}.png'.format(model_name))

    return