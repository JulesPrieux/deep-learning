from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def frac_diff(x, d):
    """
    Fractionally difference time series

    :param x: numeric vector or univariate time series
    :param d: number specifying the fractional difference order.
    :return: fractionally differenced series
    """
    if np.isnan(np.sum(x)):
        return None

    n = len(x)
    if n < 2:
        return None

    x = np.subtract(x, np.mean(x))

    # calculate weights
    weights = [0] * n
    weights[0] = -d
    for k in range(2, n):
        weights[k - 1] = weights[k - 2] * (k - 1 - d) / k

    # difference series
    ydiff = list(x)

    for i in range(0, n):
        dat = x[:i]
        w = weights[:i]
        ydiff[i] = x[i] + np.dot(w, dat[::-1])

    return ydiff


def optimal_frac_differencing(
    series: pd.Series, differences_range: Iterable
) -> pd.Series:
    """
    Finds the optimal fractional differencing value that minimizes the p-value of the ADF test.

    Parameters:
    series (pd.Series): The time series to be fractionally differenced.
    differences_range (Iterable): An iterable containing the range of fractional differencing values to test.

    Returns:
    pd.Series: A series of p-values indexed by the fractional differencing values.
    """

    def get_pvalue(series, d) -> float:
        """
        Computes the p-value of the ADF test for a given fractional differencing value.

        Parameters:
        series (pd.Series): The time series to be fractionally differenced.
        d (float): The order of differencing.

        Returns:
        float: The p-value of the ADF test.
        """
        fractioned_series = frac_diff(series, d)
        return adfuller(fractioned_series)[1]

    pvalues = [get_pvalue(series, d) for d in differences_range]
    return pd.Series(pvalues, index=differences_range)
