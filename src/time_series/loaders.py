import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from TSB_AD.utils.slidingWindows import find_length_rank
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from src.utils.logging import logger
from statsmodels.tsa.stattools import acf


def find_optimal_window_size(
    train_series: np.ndarray, min_window: int = 10, max_window: int = 100
) -> int:
    n = len(train_series)
    max_lag = min(max_window, n // 4)

    try:
        # Compute ACF
        acf_values = acf(train_series, nlags=max_lag, fft=True)

        # Confidence interval threshold (95%)
        threshold = 1.96 / np.sqrt(n)

        # Find first significant drop
        for lag in range(1, len(acf_values)):
            if abs(acf_values[lag]) < threshold:
                window_size = lag
                break
        else:
            # Fallback: use 1/4 of series length
            window_size = max_lag // 2

        # Clip to allowed range
        window_size = np.clip(window_size, min_window, max_window)

        return int(window_size)

    except Exception as e:
        print(f"ACF failed: {e}. Using default window size.")
        return min(20, max_lag)


@dataclass
class TSBFileInfo:
    dataset: str
    train_index: int


def parse_tsb_file_path(file_path: Path) -> TSBFileInfo:
    name = file_path.stem
    parts = name.split("_")

    dataset = parts[1]
    train_index = int(parts[parts.index("tr") + 1])

    return TSBFileInfo(dataset, train_index)


def load_tsb_ad_u(file_path: Path):
    file_info = parse_tsb_file_path(file_path)

    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, :-1].values.astype(float)
    labels = df.iloc[:, -1].astype(int).to_numpy()

    X_train = data[: file_info.train_index].copy()
    X_test = data[file_info.train_index :].copy()

    y_train = labels[: file_info.train_index].copy()
    y_test = labels[file_info.train_index :].copy()

    # Calculated according to the TSB-AD convention. Reference:
    # https://github.com/TheDATUMorg/TSB-AD/blob/main/benchmark_exp/Run_Detector_U.py#L62
    slidingWindow_TSB_AD_evaluation = int(
        find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
    )

    window_len = find_optimal_window_size(
        data[: file_info.train_index, 0].reshape(-1, 1)
    )
    stride = window_len
    logger.info(f"Found optimal {window_len=}")

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    rv = {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train,
        "y_test": y_test,
        "slidingWindow_TSB_AD_evaluation": slidingWindow_TSB_AD_evaluation,
        "window_len": window_len,
        "stride": stride,
        "length_T": int(data.shape[0]),
        "width_D": int(data.shape[1]),
        "dataset": file_info.dataset,
    }

    return rv
