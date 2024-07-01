import pandas as pd


def triple_barrier_labeling(
    prices: pd.Series, profit_taking: float, stop_loss: float, time_barrier: int
) -> pd.Series:
    """
    Apply the Triple-Barrier Labeling method to the given price series.

    Parameters:
    prices (pd.Series): The price series.
    profit_taking (float): The profit taking threshold as a percentage.
    stop_loss (float): The stop loss threshold as a percentage.
    time_barrier (int): The time barrier in terms of number of periods.

    Returns:
    pd.Series: A series of labels (1, -1, 0) with the same index as the input prices.
    """
    labels = pd.Series(index=prices.index, dtype=int)

    for idx in prices.index:
        entry_price = prices[idx]
        profit_target = entry_price * (1 + profit_taking)
        stop_loss_target = entry_price * (1 + stop_loss)
        time_barrier_idx = min(idx + time_barrier, prices.index[-1])

        for i in prices.index[
            prices.index.get_indexer([idx], method="nearest")[
                0
            ] : prices.index.get_indexer([time_barrier_idx], method="nearest")[0]
            + 1
        ]:
            if prices[i] >= profit_target:
                labels[idx] = 1
                break
            elif prices[i] <= stop_loss_target:
                labels[idx] = -1
                break
            elif i == time_barrier_idx:
                labels[idx] = 0
                break

    return labels
