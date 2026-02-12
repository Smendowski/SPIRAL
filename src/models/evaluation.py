from TSB_AD.evaluation.metrics import get_metrics
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utils.logging import logger
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def evaluate_tsb_ad_nonoverlapping(
    model: torch.nn.Module,
    test_loader: DataLoader,
    y_test: np.ndarray,
    window_len: int,
    slidingWindow: int,
    device: str,
    output_dir: str = None,
    model_name: str = None,
):
    model.eval()

    # Map order of images to be compliant with the time-based order
    ds = test_loader.dataset
    # ImageFolder.samples = [(path, class_idx), ...]
    temporal_indices = []
    for filepath, _ in ds.samples:
        # sample_00000042.png -> 42
        fname = Path(filepath).stem
        true_idx = int(fname.split("_")[-1])
        temporal_indices.append(true_idx)
    temporal_indices = np.array(temporal_indices)
    sort_order = np.argsort(temporal_indices)  # ImageFolder order -> temporal order

    # Order of inference in Image Folder - first class 0, then class 1
    window_scores = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_rec = model(x)
            err = ((x - x_rec) ** 2).mean(dim=(1, 2, 3))
            window_scores.append(err.cpu().numpy())

    window_scores = np.concatenate(window_scores)

    # Sort scores according to the time-based order to match time-based order of true labels
    window_scores = window_scores[sort_order]

    # Map window score to point score with optional padding
    point_scores = np.repeat(window_scores, window_len)
    point_scores = point_scores[: len(y_test)]
    if len(point_scores) < len(y_test):
        point_scores = np.pad(
            point_scores, (0, len(y_test) - len(point_scores)), mode="edge"
        )

    # Scalce scores as done in TSB-AD benchmark
    scaler = MinMaxScaler(feature_range=(0, 1))
    point_scores = scaler.fit_transform(point_scores.reshape(-1, 1)).ravel()

    normal_scores = point_scores[y_test == 0]
    anomaly_scores = point_scores[y_test == 1]
    logger.info(
        f"Normal ({len(normal_scores)}) reconstruction error: mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}"
    )
    logger.info(
        f"Anomaly ({len(anomaly_scores)}) reconstruction error: mean={anomaly_scores.mean():.4f}, std={anomaly_scores.std():.4f}"
    )
    logger.info(
        f"Overlap? Normal max={normal_scores.max():.4f}, Anomaly min={anomaly_scores.min():.4f}"
    )

    # Use original function from TSB-AD benchmark
    metrics = get_metrics(
        score=point_scores.astype(np.float64),
        labels=y_test.astype(np.int32),
        slidingWindow=slidingWindow,
        pred=None,
        version="opt",
        thre=250,
    )
    logger.info(metrics)

    return metrics
