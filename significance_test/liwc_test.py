import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

try:
    from .utils import _run_corrected_ttest
except ImportError:
    from utils import _run_corrected_ttest

LIWC_CATE = [
    "function",
    "pronoun",
    "ppron",
    "i",
    "we",
    "you",
    "shehe",
    "they",
    "ipron",
    "article",
    "prep",
    "auxverb",
    "adverb",
    "conj",
    "negate",
    "verb",
    "adj",
    "compare",
    "interrog",
    "number",
    "quant",
    "affect",
    "posemo",
    "negemo",
    "anx",
    "anger",
    "sad",
    "social",
    "family",
    "friend",
    "female",
    "male",
    "cogproc",
    "insight",
    "cause",
    "discrep",
    "tentat",
    "certain",
    "differ",
    "percept",
    "see",
    "hear",
    "feel",
    "bio",
    "body",
    "health",
    "sexual",
    "ingest",
    "drives",
    "affiliation",
    "achiev",
    "power",
    "reward",
    "risk",
    "focuspast",
    "focuspresent",
    "focusfuture",
    "relativ",
    "motion",
    "space",
    "time",
    "work",
    "leisure",
    "home",
    "money",
    "relig",
    "death",
    "informal",
    "swear",
    "netspeak",
    "assent",
    "nonflu",
    "filler",
]


def run_test(
    liwc_file=os.path.join(os.getcwd(), "output", "LIWC_feature.pkl"),
    output_path=os.path.join(os.getcwd(), "output", "sigtest"),
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(liwc_file, "rb") as f:
        all_data = pickle.load(f)
    with open(liwc_file.rsplit(".", maxsplit=1)[0] + "_files.txt", "r") as f:
        files = f.readlines()
    files = [i.strip() for i in files]
    assert len(files) == len(all_data)

    truthful_data, deception_data = [], []
    for idx in range(len(all_data)):
        data = all_data[idx, :]
        file = files[idx]

        # 归一化
        data = data / (data.sum() + 1e-6)

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
    for idx in range(truthful_data.shape[0]):
        t_25 = np.nanpercentile(truthful_data[idx], 25)
        t_75 = np.nanpercentile(truthful_data[idx], 75)
        d_25 = np.nanpercentile(deception_data[idx], 25)
        d_75 = np.nanpercentile(deception_data[idx], 75)

        truthful_data_clean.append(
            truthful_data[idx][
                (t_25 <= truthful_data[idx]) & (truthful_data[idx] <= t_75)
            ]
        )
        deception_data_clean.append(
            deception_data[idx][
                (d_25 <= deception_data[idx]) & (deception_data[idx] <= d_75)
            ]
        )

    # 数据可视化-直方图
    n_col = 10
    fig, ax = plt.subplots(
        int((len(LIWC_CATE) - 0.1) // n_col + 1), n_col, sharey="row"
    )
    fig.set_figheight(30)
    fig.set_figwidth(40)
    fig.set_dpi(20)
    fig.tight_layout()
    for i in range(0, len(LIWC_CATE), n_col):
        for j in range(n_col):
            if (i + j) >= len(LIWC_CATE):
                break
            ax[i // n_col][j].hist(
                [truthful_data_clean[i + j], deception_data_clean[i + j]],
                label=["Truthful", "Deceptive"],
            )
            ax[i // n_col][j].legend()
            ax[i // n_col][j].set(title=f"{LIWC_CATE[i + j]}", ylabel="num")
    plt.savefig(os.path.join(output_path, "liwc_dis_hist.png"), dpi=200)
    # 数据可视化-小提琴图
    n_col = 10
    fig, ax = plt.subplots(
        int((len(LIWC_CATE) - 0.1) // n_col + 1), n_col, sharey="row"
    )
    fig.set_figheight(30)
    fig.set_figwidth(40)
    fig.set_dpi(20)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    for i in range(0, len(LIWC_CATE), n_col):
        for j in range(n_col):
            if (i + j) >= len(LIWC_CATE):
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
            ax[i // n_col][j].set(title=f"{LIWC_CATE[i + j]}", ylabel="num")
    plt.savefig(os.path.join(output_path, "liwc_dis_violin.png"), dpi=200)

    statistics, pvalues = [], []
    for idx in range(len(truthful_data_clean)):
        statistic, pvalue = _run_corrected_ttest(
            truthful_data_clean[idx], deception_data_clean[idx]
        )
        pvalues.append(pvalue)
        statistics.append(statistic)
    pvalues = np.array(pvalues)
    statistics = np.array(statistics)

    diff = 1 - pvalues[None, :]
    plt.figure(figsize=(35, 8), dpi=300)
    plt.imshow(diff, cmap="OrRd")
    plt.xticks(list(range(len(LIWC_CATE))), LIWC_CATE, rotation=270)
    plt.yticks([], [])
    plt.title("1 - pvalue")
    plt.colorbar(orientation="horizontal", fraction=0.05)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "liwc_pvalue.png"), dpi=300, bbox_inches="tight"
    )

    plt.figure(figsize=(35, 8), dpi=300)
    plt.imshow(np.abs(statistics)[None, :], cmap="OrRd")
    plt.xticks(list(range(len(LIWC_CATE))), LIWC_CATE, rotation=270)
    plt.yticks([], [])
    plt.title("Statistics")
    plt.colorbar(orientation="horizontal", fraction=0.05)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "liwc_statistics.png"), dpi=300, bbox_inches="tight"
    )

    result = dict(
        liwc_pvalues=pvalues.tolist(),
        liwc_statistics=statistics.tolist(),
        liwc_items=LIWC_CATE,
    )

    return result


if __name__ == "__main__":
    run_test()
