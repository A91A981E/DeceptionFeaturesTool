import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from itertools import product
import pickle
import pandas as pd
import seaborn as sns

try:
    from .utils import _run_corrected_ttest
except ImportError:
    from utils import _run_corrected_ttest


def run_test(
    mfcc_file=os.path.join(os.getcwd(), "output", "MFCC_output.pkl"),
    output_path=os.path.join(os.getcwd(), "output", "sigtest"),
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 读取文件
    with open(mfcc_file, "rb") as f:
        all_data = pickle.load(f)
    with open(mfcc_file.rsplit(".", maxsplit=1)[0] + "_files.txt", "r") as f:
        files = f.readlines()
    files = [i.strip() for i in files]
    assert len(files) == len(all_data)

    truthful_data, deception_data = [], []
    for idx, _ in enumerate(all_data):
        data = all_data[idx]
        file = files[idx]

        data_maximum = np.nanmax(data, axis=1)
        data_minimum = np.nanmin(data, axis=1)
        data_mean = np.nanmean(data, axis=1)
        data_median = np.nanmedian(data, axis=1)
        data_std = np.nanstd(data, axis=1)
        data_var = np.nanvar(data, axis=1)
        data_kurt = scipy.stats.kurtosis(data, axis=1)
        data_skew = scipy.stats.skew(data, axis=1)
        data_percentile25 = np.nanpercentile(data, 25, axis=1)
        data_percentile50 = np.nanpercentile(data, 50, axis=1)
        data_percentile75 = np.nanpercentile(data, 75, axis=1)

        data = np.concatenate(
            list(
                zip(
                    data_maximum,
                    data_minimum,
                    data_mean,
                    data_median,
                    data_std,
                    data_var,
                    data_kurt,
                    data_skew,
                    data_percentile25,
                    data_percentile50,
                    data_percentile75,
                )
            ),
            axis=0,
        )

        if "Deceptive" in file and "Truthful" in file:
            raise RuntimeError
        if "Deceptive" in file:
            deception_data.append(data)
        elif "Truthful" in file:
            truthful_data.append(data)
        else:
            raise KeyError
    truthful_data = np.stack(truthful_data).T
    deception_data = np.stack(deception_data).T

    truthful_data_clean = []
    deception_data_clean = []
    for i in range(truthful_data.shape[0]):
        t_25 = np.nanpercentile(truthful_data[i, :], 25)
        t_75 = np.nanpercentile(truthful_data[i, :], 75)
        d_25 = np.nanpercentile(deception_data[i, :], 25)
        d_75 = np.nanpercentile(deception_data[i, :], 75)

        truthful_data_clean.append(
            truthful_data[i, :][
                (t_25 <= truthful_data[i, :]) & (truthful_data[i, :] <= t_75)
            ]
        )
        deception_data_clean.append(
            deception_data[i, :][
                (d_25 <= deception_data[i, :]) & (deception_data[i, :] <= d_75)
            ]
        )

    columns = np.array([f"mfcc_d{i+1}" for i in range(39)])  # 39 is the origin n_mfcc
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

    # 数据可视化-直方图
    n_col = 11
    fig, ax = plt.subplots(int((len(names) - 0.1) // n_col + 1), n_col, sharey="row")
    fig.set_figheight(100)
    fig.set_figwidth(80)
    fig.set_dpi(200)
    fig.tight_layout()
    for i in range(0, len(names), n_col):
        for j in range(n_col):
            if (i + j) >= len(names):
                break
            ax[i // n_col][j].hist(
                [truthful_data_clean[i + j], deception_data_clean[i + j]],
                label=["Truthful", "Deceptive"],
            )
            ax[i // n_col][j].legend()
            ax[i // n_col][j].set(title=f"{names[i + j]}", ylabel="num")
    plt.savefig(os.path.join(output_path, "mfcc_openmm_dis_hist.png"), dpi=200)
    # 数据可视化-小提琴图
    n_col = 11
    fig, ax = plt.subplots(int((len(names) - 0.1) // n_col + 1), n_col, sharey="row")
    fig.set_figheight(100)
    fig.set_figwidth(80)
    fig.set_dpi(200)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    for i in range(0, len(names), n_col):
        for j in range(n_col):
            if (i + j) >= len(names):
                break
            df = pd.DataFrame(
                {
                    "Label": ["Truthful"] * len(truthful_data_clean[i + j])
                    + ["Deceptive"] * len(deception_data_clean[i + j]),
                    "Data": truthful_data_clean[i + j].tolist()
                    + deception_data_clean[i + j].tolist(),
                    "All": [""]
                    * (
                        len(truthful_data_clean[i + j])
                        + len(deception_data_clean[i + j])
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
            ax[i // n_col][j].set(title=f"{names[i + j]}", ylabel="num")
    plt.savefig(os.path.join(output_path, "mfcc_openmm_dis_violin.png"), dpi=200)

    # 假设检验
    p_values = []
    statistics = []
    for i in range(len(truthful_data_clean)):
        statistic, p_value = _run_corrected_ttest(
            truthful_data_clean[i], deception_data_clean[i]
        )
        p_values.append(p_value)
        statistics.append(statistic)

    p_values = np.array(p_values)
    statistics = np.array(statistics)

    diff = 1 - p_values[None, :]
    diff = diff.reshape(-1, 11)
    plt.figure(figsize=(6, 12), dpi=200)
    plt.imshow(diff, cmap="OrRd")
    plt.xticks(list(range(len(names1))), names1, rotation=270)
    plt.yticks(list(range(len(columns))), columns)
    plt.title("1 - pvalue")
    plt.colorbar(fraction=0.07)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "mfcc_openmm_pvalue.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(6, 12), dpi=200)
    plt.imshow(np.abs(statistics).reshape(-1, 11), cmap="OrRd")
    plt.xticks(list(range(len(names1))), names1, rotation=270)
    plt.yticks(list(range(len(columns))), columns)
    plt.title("Statistics")
    plt.colorbar(fraction=0.07)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "mfcc_openmm_statistics.png"),
        dpi=200,
        bbox_inches="tight",
    )

    result = dict(
        mfcc_openmm_ttest=p_values.tolist(),
        mfcc_openmm_statistics=statistics.tolist(),
        mfcc_openmm_ttest_items=names.tolist(),
    )
    return result


if __name__ == "__main__":
    run_test()
