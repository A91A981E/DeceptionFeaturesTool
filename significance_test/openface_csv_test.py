import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ttest_ind, levene
import scipy
from itertools import product

try:
    from .utils import _run_corrected_ttest
except ImportError:
    from utils import _run_corrected_ttest


def get_blink_ttest(csv_files, output_path):
    # 统计全部文件
    truthful_left_openmm = []
    truthful_right_openmm = []
    deceptive_left_openmm = []
    deceptive_right_openmm = []
    for file in tqdm(csv_files):
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df = df[df["success"] == 1]
        if len(df) == 0:
            continue
        eye_lmk_11 = pd.concat(
            [
                df.loc[:, "eye_lmk_X_11"],
                df.loc[:, "eye_lmk_Y_11"],
                df.loc[:, "eye_lmk_Z_11"],
            ],
            axis=1,
        )
        eye_lmk_17 = pd.concat(
            [
                df.loc[:, "eye_lmk_X_17"],
                df.loc[:, "eye_lmk_Y_17"],
                df.loc[:, "eye_lmk_Z_17"],
            ],
            axis=1,
        )
        eye_lmk_39 = pd.concat(
            [
                df.loc[:, "eye_lmk_X_39"],
                df.loc[:, "eye_lmk_Y_39"],
                df.loc[:, "eye_lmk_Z_39"],
            ],
            axis=1,
        )
        eye_lmk_45 = pd.concat(
            [
                df.loc[:, "eye_lmk_X_45"],
                df.loc[:, "eye_lmk_Y_45"],
                df.loc[:, "eye_lmk_Z_45"],
            ],
            axis=1,
        )

        left = np.linalg.norm(eye_lmk_11.to_numpy() - eye_lmk_17.to_numpy(), axis=1)
        right = np.linalg.norm(eye_lmk_39.to_numpy() - eye_lmk_45.to_numpy(), axis=1)

        left_maximum = np.nanmax(left)
        left_minimum = np.nanmin(left)
        left_mean = np.nanmean(left)
        left_median = np.nanmedian(left)
        left_std = np.nanstd(left)
        left_var = np.nanvar(left)
        left_kurt = scipy.stats.kurtosis(left)
        left_skew = scipy.stats.skew(left)
        left_percentile25 = np.nanpercentile(left, 25)
        left_percentile50 = np.nanpercentile(left, 50)
        left_percentile75 = np.nanpercentile(left, 75)

        left_openmm = np.array(
            [
                left_maximum,
                left_minimum,
                left_mean,
                left_median,
                left_std,
                left_var,
                left_kurt,
                left_skew,
                left_percentile25,
                left_percentile50,
                left_percentile75,
            ]
        )  # (max, min, ...)

        right_maximum = np.nanmax(right)
        right_minimum = np.nanmin(right)
        right_mean = np.nanmean(right)
        right_median = np.nanmedian(right)
        right_std = np.nanstd(right)
        right_var = np.nanvar(right)
        right_kurt = scipy.stats.kurtosis(right)
        right_skew = scipy.stats.skew(right)
        right_percentile25 = np.nanpercentile(right, 25)
        right_percentile50 = np.nanpercentile(right, 50)
        right_percentile75 = np.nanpercentile(right, 75)

        right_openmm = np.array(
            [
                right_maximum,
                right_minimum,
                right_mean,
                right_median,
                right_std,
                right_var,
                right_kurt,
                right_skew,
                right_percentile25,
                right_percentile50,
                right_percentile75,
            ]
        )  # (max, min, ...)

        if "Deceptive" in file and "Truthful" in file:
            raise RuntimeError
        if "Deceptive" in file:
            deceptive_left_openmm.append(left_openmm)
            deceptive_right_openmm.append(right_openmm)
        elif "Truthful" in file:
            truthful_left_openmm.append(left_openmm)
            truthful_right_openmm.append(right_openmm)
        else:
            raise KeyError
    truthful_left_openmm = np.array(truthful_left_openmm)
    truthful_right_openmm = np.array(truthful_right_openmm)
    deceptive_left_openmm = np.array(deceptive_left_openmm)
    deceptive_right_openmm = np.array(deceptive_right_openmm)

    # 去除离群点（四分位数）
    truthful_left_openmm_gen = []
    truthful_right_openmm_gen = []
    deceptive_left_openmm_gen = []
    deceptive_right_openmm_gen = []
    for i in range(truthful_left_openmm.shape[-1]):
        tl_25 = np.nanpercentile(truthful_left_openmm[:, i], 25)
        tl_75 = np.nanpercentile(truthful_left_openmm[:, i], 75)
        dl_25 = np.nanpercentile(deceptive_left_openmm[:, i], 25)
        dl_75 = np.nanpercentile(deceptive_left_openmm[:, i], 75)
        tr_25 = np.nanpercentile(truthful_right_openmm[:, i], 25)
        tr_75 = np.nanpercentile(truthful_right_openmm[:, i], 75)
        dr_25 = np.nanpercentile(deceptive_right_openmm[:, i], 25)
        dr_75 = np.nanpercentile(deceptive_right_openmm[:, i], 75)

        truthful_left_openmm_gen.append(
            truthful_left_openmm[:, i][
                (tl_25 <= truthful_left_openmm[:, i])
                & (truthful_left_openmm[:, i] <= tl_75)
            ]
        )
        truthful_right_openmm_gen.append(
            truthful_right_openmm[:, i][
                (tr_25 <= truthful_right_openmm[:, i])
                & (truthful_right_openmm[:, i] <= tr_75)
            ]
        )
        deceptive_left_openmm_gen.append(
            deceptive_left_openmm[:, i][
                (dl_25 <= deceptive_left_openmm[:, i])
                & (deceptive_left_openmm[:, i] <= dl_75)
            ]
        )
        deceptive_right_openmm_gen.append(
            deceptive_right_openmm[:, i][
                (dr_25 <= deceptive_right_openmm[:, i])
                & (deceptive_right_openmm[:, i] <= dr_75)
            ]
        )

    del truthful_left_openmm, truthful_right_openmm
    del deceptive_left_openmm, deceptive_right_openmm

    names = np.array(
        [
            "max",
            "min",
            "mean",
            "median",
            "std",
            "var",
            "kurt",
            "skew",
            "percentile25",
            "percentile50",
            "percentile75",
        ]
    )

    # 数据可视化-小提琴图
    fig, ax = plt.subplots(len(names), 2, sharey="row")
    fig.set_figheight(30)
    fig.set_figwidth(10)
    fig.set_dpi(20)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    for i in range(len(names)):
        # left
        df = pd.DataFrame(
            {
                "Label": ["Truthful"] * len(truthful_left_openmm_gen[i])
                + ["Deceptive"] * len(deceptive_left_openmm_gen[i]),
                "Data": truthful_left_openmm_gen[i].tolist()
                + deceptive_left_openmm_gen[i].tolist(),
                "All": [""]
                * (
                    len(truthful_left_openmm_gen[i]) + len(deceptive_left_openmm_gen[i])
                ),
            }
        )
        sns.violinplot(
            x="Data",
            y="All",
            hue="Label",
            data=df,
            split=True,
            ax=ax[i][0],
            inner="box",
            palette="muted",
            orient="horizontal",
        )
        ax[i][0].set(title=f"Left eye blink {names[i]}", ylabel="")

        # right
        df = pd.DataFrame(
            {
                "Label": ["Truthful"] * len(truthful_right_openmm_gen[i])
                + ["Deceptive"] * len(deceptive_right_openmm_gen[i]),
                "Data": truthful_right_openmm_gen[i].tolist()
                + deceptive_right_openmm_gen[i].tolist(),
                "All": [""]
                * (
                    len(truthful_right_openmm_gen[i])
                    + len(deceptive_right_openmm_gen[i])
                ),
            }
        )
        sns.violinplot(
            x="Data",
            y="All",
            hue="Label",
            data=df,
            split=True,
            ax=ax[i][1],
            inner="box",
            palette="muted",
            orient="horizontal",
        )
        ax[i][1].set(title=f"Right eye blink {names[i]}", ylabel="")
    plt.savefig(os.path.join(output_path, "blink_openmm_dis_violin.png"), dpi=200)
    # 数据可视化-直方图
    fig, ax = plt.subplots(len(names), 2, sharey="row")
    fig.set_figheight(30)
    fig.set_figwidth(10)
    fig.set_dpi(20)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    for i in range(len(names)):
        # left
        ax[i][0].hist(
            [truthful_left_openmm_gen[i], deceptive_left_openmm_gen[i]],
            label=["Truthful", "Deceptive"],
        )
        ax[i][0].legend()
        ax[i][0].set(title=f"Left eye blink {names[i]}", ylabel="")

        # right
        ax[i][1].hist(
            [truthful_right_openmm_gen[i], deceptive_right_openmm_gen[i]],
            label=["Truthful", "Deceptive"],
        )
        ax[i][1].legend()
        ax[i][1].set(title=f"Right eye blink {names[i]}", ylabel="")
    plt.savefig(os.path.join(output_path, "blink_openmm_dis_hist.png"), dpi=200)

    # 假设检验
    ttest_left_pvalue, ttest_left_statistics = [], []
    ttest_right_pvalue, ttest_right_statistics = [], []
    for i in range(len(truthful_left_openmm_gen)):
        statistic, p_value = _run_corrected_ttest(
            truthful_left_openmm_gen[i], deceptive_left_openmm_gen[i]
        )
        ttest_left_pvalue.append(p_value)
        ttest_left_statistics.append(statistic)
    for i in range(len(truthful_right_openmm_gen)):
        statistic, p_value = _run_corrected_ttest(
            truthful_right_openmm_gen[i], deceptive_right_openmm_gen[i]
        )
        ttest_right_pvalue.append(p_value)
        ttest_right_statistics.append(statistic)

    ttest_left_pvalue = np.array(ttest_left_pvalue)
    ttest_right_pvalue = np.array(ttest_right_pvalue)
    ttest_left_statistics = np.array(ttest_left_statistics)
    ttest_right_statistics = np.array(ttest_right_statistics)

    diff = 1 - np.concatenate(
        [ttest_left_pvalue[None, :], ttest_right_pvalue[None, :]], axis=0
    )
    plt.figure(figsize=(10, 6), dpi=200)
    plt.tight_layout()
    plt.imshow(diff, cmap="OrRd")
    plt.yticks([0, 1], ["left", "right"])
    plt.xticks(list(range(len(names))), names, rotation=270)
    plt.title("1 - pvalue")
    # plt.colorbar()
    plt.colorbar(orientation="horizontal", fraction=0.05, pad=0.25)
    plt.savefig(
        os.path.join(output_path, "blink_openmm_pvalue.png"),
        dpi=200,
        bbox_inches="tight",
    )

    x = np.concatenate(
        [ttest_left_statistics[None, :], ttest_right_statistics[None, :]], axis=0
    )
    plt.figure(figsize=(10, 6), dpi=200)
    plt.tight_layout()
    plt.imshow(np.abs(x), cmap="OrRd")
    plt.yticks([0, 1], ["left", "right"])
    plt.xticks(list(range(len(names))), names, rotation=270)
    plt.title("Statistics")
    # plt.colorbar()
    plt.colorbar(orientation="horizontal", fraction=0.05, pad=0.25)
    plt.savefig(
        os.path.join(output_path, "blink_openmm_statistics.png"),
        dpi=200,
        bbox_inches="tight",
    )

    result = dict(
        openface_blink_openmm_left_pvalues=ttest_left_pvalue.tolist(),
        openface_blink_openmm_right_pvalues=ttest_right_pvalue.tolist(),
        openface_blink_openmm_left_statistics=ttest_left_statistics.tolist(),
        openface_blink_openmm_right_statistics=ttest_right_statistics.tolist(),
        openface_blink_openmm_items=names.tolist(),
    )

    return result


def get_eye_relative_pos_ttest(csv_files, output_path):
    # 统计全部文件
    truthful_left = []
    truthful_right = []
    deceptive_left = []
    deceptive_right = []
    for file in tqdm(csv_files):
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df = df[df["success"] == 1]
        if len(df) == 0:
            continue
        eye_lmk_outer_left_X = df.loc[:, "eye_lmk_X_8":"eye_lmk_X_19"]
        eye_lmk_outer_left_Y = df.loc[:, "eye_lmk_Y_8":"eye_lmk_Y_19"]
        eye_lmk_outer_left_Z = df.loc[:, "eye_lmk_Z_8":"eye_lmk_Z_19"]
        eye_lmk_outer_left = np.array(
            list(
                zip(
                    eye_lmk_outer_left_X.to_numpy(),
                    eye_lmk_outer_left_Y.to_numpy(),
                    eye_lmk_outer_left_Z.to_numpy(),
                )
            )
        ).transpose(
            (0, 2, 1)
        )  # (num_frame, 12, 3)
        eye_lmk_outer_left_center = eye_lmk_outer_left.sum(axis=1) / 12  # 左眼外圈

        eye_lmk_inner_left_X = df.loc[:, "eye_lmk_X_0":"eye_lmk_X_7"]
        eye_lmk_inner_left_Y = df.loc[:, "eye_lmk_Y_0":"eye_lmk_Y_7"]
        eye_lmk_inner_left_Z = df.loc[:, "eye_lmk_Z_0":"eye_lmk_Z_7"]
        eye_lmk_inner_left = np.array(
            list(
                zip(
                    eye_lmk_inner_left_X.to_numpy(),
                    eye_lmk_inner_left_Y.to_numpy(),
                    eye_lmk_inner_left_Z.to_numpy(),
                )
            )
        ).transpose(
            (0, 2, 1)
        )  # (num_frame, 8, 3)
        eye_lmk_inner_left_center = eye_lmk_inner_left.sum(axis=1) / 8  # 左眼内圈

        eye_lmk_outer_right_X = df.loc[:, "eye_lmk_X_36":"eye_lmk_X_47"]
        eye_lmk_outer_right_Y = df.loc[:, "eye_lmk_Y_36":"eye_lmk_Y_47"]
        eye_lmk_outer_right_Z = df.loc[:, "eye_lmk_Z_36":"eye_lmk_Z_47"]
        eye_lmk_outer_right = np.array(
            list(
                zip(
                    eye_lmk_outer_right_X.to_numpy(),
                    eye_lmk_outer_right_Y.to_numpy(),
                    eye_lmk_outer_right_Z.to_numpy(),
                )
            )
        ).transpose(
            (0, 2, 1)
        )  # (num_frame, 12, 3)
        eye_lmk_outer_right_center = eye_lmk_outer_right.sum(axis=1) / 12  # 右眼外圈

        eye_lmk_inner_right_X = df.loc[:, "eye_lmk_X_28":"eye_lmk_X_35"]
        eye_lmk_inner_right_Y = df.loc[:, "eye_lmk_Y_28":"eye_lmk_Y_35"]
        eye_lmk_inner_right_Z = df.loc[:, "eye_lmk_Z_28":"eye_lmk_Z_35"]
        eye_lmk_inner_right = np.array(
            list(
                zip(
                    eye_lmk_inner_right_X.to_numpy(),
                    eye_lmk_inner_right_Y.to_numpy(),
                    eye_lmk_inner_right_Z.to_numpy(),
                )
            )
        ).transpose(
            (0, 2, 1)
        )  # (num_frame, 8, 3)
        eye_lmk_inner_right_center = eye_lmk_inner_right.sum(axis=1) / 8  # 右眼内圈

        left = eye_lmk_outer_left_center - eye_lmk_inner_left_center
        right = eye_lmk_outer_right_center - eye_lmk_inner_right_center

        left = left / np.linalg.norm(left, axis=1)[:, None]  # (num_frame, 3)
        right = right / np.linalg.norm(right, axis=1)[:, None]  # (num_frame, 3)

        left_maximum = np.nanmax(left, axis=0)
        left_minimum = np.nanmin(left, axis=0)
        left_mean = np.nanmean(left, axis=0)
        left_median = np.nanmedian(left, axis=0)
        left_std = np.nanstd(left, axis=0)
        left_var = np.nanvar(left, axis=0)
        left_kurt = scipy.stats.kurtosis(left, axis=0)
        left_skew = scipy.stats.skew(left, axis=0)
        left_percentile25 = np.nanpercentile(left, 25, axis=0)
        left_percentile50 = np.nanpercentile(left, 50, axis=0)
        left_percentile75 = np.nanpercentile(left, 75, axis=0)

        left = np.concatenate(
            list(
                zip(
                    left_maximum,
                    left_minimum,
                    left_mean,
                    left_median,
                    left_std,
                    left_var,
                    left_kurt,
                    left_skew,
                    left_percentile25,
                    left_percentile50,
                    left_percentile75,
                )
            ),
            axis=0,
        )  # (x(max, min, ...), y(max, min, ...), z(max, min, ...))

        right_maximum = np.nanmax(right, axis=0)
        right_minimum = np.nanmin(right, axis=0)
        right_mean = np.nanmean(right, axis=0)
        right_median = np.nanmedian(right, axis=0)
        right_std = np.nanstd(right, axis=0)
        right_var = np.nanvar(right, axis=0)
        right_kurt = scipy.stats.kurtosis(right, axis=0)
        right_skew = scipy.stats.skew(right, axis=0)
        right_percentile25 = np.nanpercentile(right, 25, axis=0)
        right_percentile50 = np.nanpercentile(right, 50, axis=0)
        right_percentile75 = np.nanpercentile(right, 75, axis=0)

        right = np.concatenate(
            list(
                zip(
                    right_maximum,
                    right_minimum,
                    right_mean,
                    right_median,
                    right_std,
                    right_var,
                    right_kurt,
                    right_skew,
                    right_percentile25,
                    right_percentile50,
                    right_percentile75,
                )
            ),
            axis=0,
        )  # (x(max, min, ...), y(max, min, ...), z(max, min, ...))

        if "lie" in file:
            deceptive_left.append(left)
            deceptive_right.append(right)
        else:
            truthful_left.append(left)
            truthful_right.append(right)
    truthful_left = np.array(truthful_left)
    truthful_right = np.array(truthful_right)
    deceptive_left = np.array(deceptive_left)
    deceptive_right = np.array(deceptive_right)

    #  去除离群点（四分位数）
    truthful_left_openmm_gen = []
    truthful_right_openmm_gen = []
    deceptive_left_openmm_gen = []
    deceptive_right_openmm_gen = []
    for i in range(truthful_left.shape[-1]):
        tl_25 = np.nanpercentile(truthful_left[:, i], 25)
        tl_75 = np.nanpercentile(truthful_left[:, i], 75)
        dl_25 = np.nanpercentile(deceptive_left[:, i], 25)
        dl_75 = np.nanpercentile(deceptive_left[:, i], 75)
        tr_25 = np.nanpercentile(truthful_right[:, i], 25)
        tr_75 = np.nanpercentile(truthful_right[:, i], 75)
        dr_25 = np.nanpercentile(deceptive_right[:, i], 25)
        dr_75 = np.nanpercentile(deceptive_right[:, i], 75)

        truthful_left_openmm_gen.append(
            truthful_left[:, i][
                (tl_25 <= truthful_left[:, i]) & (truthful_left[:, i] <= tl_75)
            ]
        )
        truthful_right_openmm_gen.append(
            truthful_right[:, i][
                (tr_25 <= truthful_right[:, i]) & (truthful_right[:, i] <= tr_75)
            ]
        )
        deceptive_left_openmm_gen.append(
            deceptive_left[:, i][
                (dl_25 <= deceptive_left[:, i]) & (deceptive_left[:, i] <= dl_75)
            ]
        )
        deceptive_right_openmm_gen.append(
            deceptive_right[:, i][
                (dr_25 <= deceptive_right[:, i]) & (deceptive_right[:, i] <= dr_75)
            ]
        )

    del truthful_left, truthful_right
    del deceptive_left, deceptive_right

    names = np.array(
        [
            "max",
            "min",
            "mean",
            "median",
            "std",
            "var",
            "kurt",
            "skew",
            "percentile25",
            "percentile50",
            "percentile75",
        ]
    )

    # 数据可视化-小提琴图
    fig, ax = plt.subplots(len(names) * 3, 2, sharey="row")
    fig.set_figheight(90)
    fig.set_figwidth(10)
    fig.set_dpi(20)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    for k, a in enumerate(["X", "Y", "Z"]):
        for i in range(len(names)):
            # left
            df = pd.DataFrame(
                {
                    "Label": ["Truthful"]
                    * len(truthful_left_openmm_gen[i + k * len(names)])
                    + ["Deceptive"]
                    * len(deceptive_left_openmm_gen[i + k * len(names)]),
                    "Data": truthful_left_openmm_gen[i + k * len(names)].tolist()
                    + deceptive_left_openmm_gen[i + k * len(names)].tolist(),
                    "All": [""]
                    * (
                        len(truthful_left_openmm_gen[i + k * len(names)])
                        + len(deceptive_left_openmm_gen[i + k * len(names)])
                    ),
                }
            )
            sns.violinplot(
                x="Data",
                y="All",
                hue="Label",
                data=df,
                split=True,
                ax=ax[i + k * len(names)][0],
                inner="box",
                palette="muted",
                orient="horizontal",
            )
            ax[i + k * len(names)][0].set(
                title=f"Left eye relative {a}_{names[i]}", ylabel=""
            )

            # right
            df = pd.DataFrame(
                {
                    "Label": ["Truthful"]
                    * len(truthful_right_openmm_gen[i + k * len(names)])
                    + ["Deceptive"]
                    * len(deceptive_right_openmm_gen[i + k * len(names)]),
                    "Data": truthful_right_openmm_gen[i + k * len(names)].tolist()
                    + deceptive_right_openmm_gen[i + k * len(names)].tolist(),
                    "All": [""]
                    * (
                        len(truthful_right_openmm_gen[i + k * len(names)])
                        + len(deceptive_right_openmm_gen[i + k * len(names)])
                    ),
                }
            )
            sns.violinplot(
                x="Data",
                y="All",
                hue="Label",
                data=df,
                split=True,
                ax=ax[i + k * len(names)][1],
                inner="box",
                palette="muted",
                orient="horizontal",
            )
            ax[i + k * len(names)][1].set(
                title=f"Right eye relative {a}_{names[i]}", ylabel=""
            )
    plt.savefig(
        os.path.join(output_path, "eye_relative_pos_openmm_dis_violin.png"), dpi=200
    )
    # 数据可视化-直方图
    fig, ax = plt.subplots(len(names) * 3, 2, sharey="row")
    fig.set_figheight(90)
    fig.set_figwidth(10)
    fig.set_dpi(200)
    fig.tight_layout()
    for k, a in enumerate(["X", "Y", "Z"]):
        for i in range(len(names)):
            # left
            ax[i + k * len(names)][0].hist(
                [
                    truthful_left_openmm_gen[i + k * len(names)],
                    deceptive_left_openmm_gen[i + k * len(names)],
                ],
                label=["Truthful", "Deceptive"],
            )
            ax[i + k * len(names)][0].legend()
            ax[i + k * len(names)][0].set(
                title=f"Left eye relative {a}_{names[i]}", ylabel="num"
            )

            # right
            ax[i + k * len(names)][1].hist(
                [
                    truthful_right_openmm_gen[i + k * len(names)],
                    deceptive_right_openmm_gen[i + k * len(names)],
                ],
                label=["Truthful", "Deceptive"],
            )
            ax[i + k * len(names)][1].legend()
            ax[i + k * len(names)][1].set(title=f"Right eye relative {a}_{names[i]}")
    plt.savefig(
        os.path.join(output_path, "eye_relative_pos_openmm_dis_hist.png"), dpi=200
    )

    # 假设检验
    ttest_left_pvalue, ttest_left_statistics = [], []
    ttest_right_pvalue, ttest_right_statistics = [], []
    for i in range(len(truthful_left_openmm_gen)):
        statistic, p_value = _run_corrected_ttest(
            truthful_left_openmm_gen[i],
            deceptive_left_openmm_gen[i],
        )
        ttest_left_pvalue.append(p_value)
        ttest_left_statistics.append(statistic)
    for i in range(len(truthful_right_openmm_gen)):
        statistic, p_value = _run_corrected_ttest(
            truthful_right_openmm_gen[i],
            deceptive_right_openmm_gen[i],
        )
        ttest_right_pvalue.append(p_value)
        ttest_right_statistics.append(statistic)

    ttest_left_pvalue = np.array(ttest_left_pvalue)
    ttest_right_pvalue = np.array(ttest_right_pvalue)
    ttest_left_statistics = np.array(ttest_left_statistics)
    ttest_right_statistics = np.array(ttest_right_statistics)

    diff = 1 - np.concatenate(
        [ttest_left_pvalue[None, :], ttest_right_pvalue[None, :]], axis=0
    )
    col = np.array(["X", "Y", "Z"])
    names = list(map(lambda x: x[0] + "_" + x[1], list(product(col, names))))
    plt.figure(figsize=(10, 6), dpi=200)
    plt.tight_layout()
    plt.imshow(diff, cmap="OrRd")
    plt.yticks([0, 1], ["left", "right"])
    plt.xticks(list(range(len(names))), names, rotation=270)
    plt.title("1 - pvalue")
    plt.colorbar(orientation="horizontal", fraction=0.05, pad=0.26)
    # plt.colorbar()
    plt.savefig(
        os.path.join(output_path, "eye_relative_pos_openmm_pvalues.png"),
        dpi=200,
        bbox_inches="tight",
    )

    x = np.concatenate(
        [ttest_left_statistics[None, :], ttest_right_statistics[None, :]], axis=0
    )
    plt.figure(figsize=(10, 6), dpi=200)
    plt.tight_layout()
    plt.imshow(np.abs(x), cmap="OrRd")
    plt.yticks([0, 1], ["left", "right"])
    plt.xticks(list(range(len(names))), names, rotation=270)
    plt.title("Statistics")
    # plt.colorbar()
    plt.colorbar(orientation="horizontal", fraction=0.05, pad=0.25)
    plt.savefig(
        os.path.join(output_path, "eye_relative_pos_openmm_statistics.png"),
        dpi=200,
        bbox_inches="tight",
    )

    result = dict(
        openface_eye_relative_pos_openmm_left_pvalues=ttest_left_pvalue.tolist(),
        openface_eye_relative_pos_openmm_right_pvalues=ttest_right_pvalue.tolist(),
        openface_eye_relative_pos_openmm_left_statistics=ttest_left_statistics.tolist(),
        openface_eye_relative_pos_openmm_right_statistics=ttest_right_statistics.tolist(),
        openface_eye_relative_pos_openmm_items=names,
    )

    return result


def get_aus_ttest(csv_files, output_path):
    # 统计全部文件
    deceptive_au = []
    truthful_au = []

    for file in tqdm(csv_files):
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df = df[df["success"] == 1]
        if len(df) == 0:
            continue
        au = df.loc[:, "AU01_r":"AU45_r"]

        au_maximum = np.nanmax(au, axis=0)
        au_minimum = np.nanmin(au, axis=0)
        au_mean = np.nanmean(au, axis=0)
        au_median = np.nanmedian(au, axis=0)
        au_std = np.nanstd(au, axis=0)
        au_var = np.nanvar(au, axis=0)
        au_kurt = scipy.stats.kurtosis(au, axis=0)
        au_skew = scipy.stats.skew(au, axis=0)
        au_percentile25 = np.nanpercentile(au, 25, axis=0)
        au_percentile50 = np.nanpercentile(au, 50, axis=0)
        au_percentile75 = np.nanpercentile(au, 75, axis=0)

        au = np.concatenate(
            list(
                zip(
                    au_maximum,
                    au_minimum,
                    au_mean,
                    au_median,
                    au_std,
                    au_var,
                    au_kurt,
                    au_skew,
                    au_percentile25,
                    au_percentile50,
                    au_percentile75,
                )
            ),
            axis=0,
        )

        if "lie" in file:
            deceptive_au.append(au)
        else:
            truthful_au.append(au)
    deceptive_au = np.array(deceptive_au)
    truthful_au = np.array(truthful_au)

    columns = np.array(list(df.loc[:, "AU01_r":"AU45_r"].columns))
    names1 = np.array(
        [
            "max",
            "min",
            "mean",
            "median",
            "std",
            "var",
            "kurt",
            "skew",
            "percentile25",
            "percentile50",
            "percentile75",
        ]
    )
    names = np.array(
        list(map(lambda x: x[0] + "_" + x[1], list(product(columns, names1))))
    )

    # 有一些全0的数据没有离群点
    # deceptive_au_openmm_gen = []
    # truthful_au_openmm_gen = []
    deceptive_au_openmm_gen = deceptive_au.T
    truthful_au_openmm_gen = truthful_au.T

    # for i in range(deceptive_au.shape[-1]):
    #     t_25 = np.nanpercentile(truthful_au[:, i], 25)
    #     t_75 = np.nanpercentile(truthful_au[:, i], 75)
    #     d_25 = np.nanpercentile(deceptive_au[:, i], 25)
    #     d_75 = np.nanpercentile(deceptive_au[:, i], 75)

    #     truthful_au_openmm_gen.append(truthful_au[:, i][(t_25 <= truthful_au[:, i]) & (truthful_au[:, i] <= t_75)])
    #     deceptive_au_openmm_gen.append(deceptive_au[:, i][(d_25 <= deceptive_au[:, i]) & (deceptive_au[:, i] <= d_75)])

    # 数据可视化
    # 数据可视化-小提琴图
    n_col = 11
    fig, ax = plt.subplots(int((len(names) - 0.1) // n_col + 1), n_col, sharey="row")
    fig.set_figheight(80)
    fig.set_figwidth(60)
    fig.set_dpi(200)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    for i in range(0, len(names), n_col):
        for j in range(n_col):
            if (i + j) >= len(names):
                break
            df = pd.DataFrame(
                {
                    "Label": ["Truthful"] * len(truthful_au_openmm_gen[i + j])
                    + ["Deceptive"] * len(deceptive_au_openmm_gen[i + j]),
                    "Data": truthful_au_openmm_gen[i + j].tolist()
                    + deceptive_au_openmm_gen[i + j].tolist(),
                    "All": [""]
                    * (
                        len(truthful_au_openmm_gen[i + j])
                        + len(deceptive_au_openmm_gen[i + j])
                    ),
                }
            )
            sns.violinplot(
                x="Data",
                y="All",
                hue="Label",
                data=df,
                split=True,
                ax=ax[i // n_col][j],
                inner="box",
                palette="muted",
                orient="horizontal",
            )
            ax[i // n_col][j].set(title=f"Action Unit {names[i + j]}", ylabel="")
    plt.savefig(os.path.join(output_path, "au_openmm_dis_violin.png"), dpi=200)
    # 数据可视化-直方图
    n_col = 11
    fig, ax = plt.subplots(int((len(names) - 0.1) // n_col + 1), n_col, sharey="row")
    fig.set_figheight(80)
    fig.set_figwidth(60)
    fig.set_dpi(200)
    fig.tight_layout()
    for i in range(0, len(names), n_col):
        for j in range(n_col):
            if (i + j) >= len(names):
                break
            ax[i // n_col][j].hist(
                [truthful_au_openmm_gen[i + j], deceptive_au_openmm_gen[i + j]],
                label=["Truthful", "Deceptive"],
            )
            ax[i // n_col][j].legend()
            ax[i // n_col][j].set(title=f"Action Unit {names[i + j]}", ylabel="num")

    plt.savefig(os.path.join(output_path, "au_openmm_dis_hist.png"), dpi=200)

    # 假设检验
    ttest_pvalue = []
    ttest_statistic = []
    for i in range(len(truthful_au_openmm_gen)):
        statistic, p_value = _run_corrected_ttest(
            truthful_au_openmm_gen[i],
            deceptive_au_openmm_gen[i],
        )
        ttest_pvalue.append(p_value)
        ttest_statistic.append(statistic)

    ttest_pvalue = np.array(ttest_pvalue)
    ttest_statistic = np.array(ttest_statistic)
    ttest_pvalue[np.isnan(ttest_pvalue)] = 1

    diff = 1 - ttest_pvalue[None, :]
    diff = diff.reshape(-1, 11)
    plt.figure(figsize=(6, 8), dpi=200)
    plt.imshow(diff, cmap="OrRd")
    plt.xticks(list(range(len(names1))), names1, rotation=270)
    plt.yticks(list(range(len(columns))), columns)
    plt.title("1 - pvalue")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "au_openmm_pvalue.png"), dpi=200, bbox_inches="tight"
    )

    x = np.abs(ttest_statistic[None, :].reshape(-1, 11))
    plt.figure(figsize=(6, 8), dpi=200)
    plt.imshow(x, cmap="OrRd")
    plt.xticks(list(range(len(names1))), names1, rotation=270)
    plt.yticks(list(range(len(columns))), columns)
    plt.title("Statistics")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "au_openmm_statistics.png"),
        dpi=200,
        bbox_inches="tight",
    )

    result = dict(
        openface_au_openmm_ttest=ttest_pvalue.tolist(),
        openface_au_openmm_ttest_statistic=ttest_statistic.tolist(),
        openface_au_items=names.tolist(),
    )

    return result


def run_test(
    csv_files=os.path.join(os.getcwd(), "output", "openface"),
    output_path=os.path.join(os.getcwd(), "output", "sigtest"),
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    csv_files = [os.path.join(csv_files, i) for i in os.listdir(csv_files)]

    csv_files = [
        os.path.join(a, i)
        for a in csv_files
        for i in os.listdir(a)
        if i.endswith(".csv")
    ]
    all_results = {}

    blink = get_blink_ttest(csv_files, output_path)
    all_results.update(blink)

    eye_relative_pos = get_eye_relative_pos_ttest(csv_files, output_path)
    all_results.update(eye_relative_pos)

    au = get_aus_ttest(csv_files, output_path)
    all_results.update(au)

    return all_results


if __name__ == "__main__":
    run_test()
