# trading-agent/src/prediction/labels.py
"""
Label generation for supervised learning.

We predict the NEXT N bars' return direction — not the price itself.

Why classify instead of regress?
- Markets have fat tails. Regression MSE punishes outliers too harshly.
- We don't need to know HOW MUCH the price moves — just direction + confidence.
- Classification gives calibrated probabilities we can use for position sizing.

Label scheme (3-class):
    BUY  = 1  : next N-bar return in top 40th percentile
    HOLD = 0  : next N-bar return in middle 20th percentile
    SELL = -1 : next N-bar return in bottom 40th percentile
"""
import numpy as np
import pandas as pd
from enum import IntEnum
from loguru import logger


class Signal(IntEnum):
    SELL = -1
    HOLD = 0
    BUY  = 1


def make_labels(
    close: pd.Series,
    horizon: int = 5,
    buy_threshold: float = 0.40,
    sell_threshold: float = 0.40,
) -> pd.Series:
    """
    Generate forward-return labels.

    Parameters
    ----------
    close          : closing price series
    horizon        : how many bars ahead to measure return
    buy_threshold  : top X% of returns → BUY label
    sell_threshold : bottom X% of returns → SELL label
    (remaining middle % → HOLD)

    CRITICAL: The last `horizon` bars will be NaN — they have no
    future return yet. These rows MUST be excluded from training.
    """
    # Forward return: percentage gain over next `horizon` bars
    fwd_return = close.shift(-horizon) / close - 1

    # Percentile cutoffs — computed on entire series (use training data only in practice)
    buy_cutoff  = fwd_return.quantile(1 - buy_threshold)
    sell_cutoff = fwd_return.quantile(sell_threshold)

    labels = pd.Series(Signal.HOLD, index=close.index, dtype=int)
    labels[fwd_return >= buy_cutoff]  = Signal.BUY
    labels[fwd_return <= sell_cutoff] = Signal.SELL
    labels[fwd_return.isna()] = np.nan   # last `horizon` bars — no label

    buy_count  = (labels == Signal.BUY).sum()
    sell_count = (labels == Signal.SELL).sum()
    hold_count = (labels == Signal.HOLD).sum()
    logger.debug(
        f"Labels (horizon={horizon}d): "
        f"BUY={buy_count} SELL={sell_count} HOLD={hold_count} NaN={labels.isna().sum()}"
    )
    return labels


def make_binary_labels(close: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Simpler 2-class version: UP=1, DOWN=0.
    Use when you want a single probability output (easier to threshold).
    """
    fwd_return = close.shift(-horizon) / close - 1
    labels = (fwd_return > 0).astype(float)
    labels[fwd_return.isna()] = np.nan
    return labels