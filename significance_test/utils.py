import numpy as np
from scipy import stats


def _run_corrected_ttest(
    data1: np.ndarray, data2: np.ndarray, thred: float = 0.05
) -> float:
    # 执行正态性检验
    _, p1 = stats.shapiro(data1)
    _, p2 = stats.shapiro(data2)

    # 执行方差齐性检验
    _, p_var = stats.levene(data1, data2)

    if p1 > thred and p2 > thred:
        # 如果数据满足正态性假设，执行独立样本 t 检验
        if p_var > thred:
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=True)
        else:
            statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    else:
        # 如果数据不满足正态性假设，使用非参数检验，如Mann-Whitney U 检验
        statistic, p_value = stats.mannwhitneyu(data1, data2)
    return statistic, p_value
