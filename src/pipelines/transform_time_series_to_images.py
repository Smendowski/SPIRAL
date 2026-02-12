import tempfile
import shutil
from pathlib import Path
import json
from src.utils.logging import logger
from src.utils.reproducibility import set_reproducibility
import warnings
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from src.time_series.loaders import load_tsb_ad_u
from src.images.creators import create_png_image_dataset
from time import perf_counter
from src.transformations.ts2i.gramian_angular_field import (
    gramian_angular_summation_field,
    gramian_angular_difference_field,
)
from src.transformations.ts2i.random_noise import random_noise
from src.transformations.ts2i.line_plot import line_plot
from src.transformations.ts2i.state_grid import state_grid
from src.transformations.ts2i.markov_transition_field import markov_transition_field
from src.transformations.ts2i.recurrence_plot import recurrence_plot
from src.transformations.ts2i.wavelets import (
    ricker_wavelet_transform,
    morlet_wavelet_transform,
)
import os
from datetime import datetime
from src.transformations.ts2i.novel import spiral


def run_pipeline(csv_file: str, output_dir: str, config_file: str, device: str) -> None:
    warnings.filterwarnings("ignore")

    logger.info(f"Loading configuration from {config_file}")
    config: DictConfig = OmegaConf.load(config_file)
    set_reproducibility(config.get("random_seed", 42))

    file_paths = pd.read_csv(csv_file)["file_name"].to_list()
    for _file_path in file_paths:
        file_path = Path("data") / "TSB-AD" / "TSB-AD-U" / _file_path
        if not file_path.exists():
            raise FileNotFoundError(
                f"Expected {_file_path} to be in the following location: {file_path}"
            )

        logger.info(f"Loading data from {file_path}")
        loader_rv = load_tsb_ad_u(file_path)
        X_train, X_test = loader_rv["X_train"], loader_rv["X_test"]
        y_train, y_test = loader_rv["y_train"], loader_rv["y_test"]

        if y_test.sum() == 0:
            logger.warning(f"Skipping {_file_path} — no anomalies in y_test")
            continue

        TS2I_TRANSFORMATIONS = [
            random_noise,
            state_grid,
            line_plot,
            gramian_angular_summation_field,
            gramian_angular_difference_field,
            markov_transition_field,
            recurrence_plot,
            ricker_wavelet_transform,
            morlet_wavelet_transform,
            spiral,
        ]

        for ts2i_transformation in TS2I_TRANSFORMATIONS:
            logger.info(
                f"Transforming {file_path.stem} using {ts2i_transformation.pretty_name}"
            )

            storage = Path(os.environ.get("PLG_GROUPS_STORAGE")) / "plggaiforsoc"
            with tempfile.TemporaryDirectory(dir=storage) as tmp_dir:
                tmp_base = Path(tmp_dir)
                train_dir = tmp_base / "train"
                test_dir = tmp_base / "test"

                (train_dir / "0").mkdir(parents=True, exist_ok=True)
                (test_dir / "0").mkdir(parents=True, exist_ok=True)
                (test_dir / "1").mkdir(parents=True, exist_ok=True)

                window_len = loader_rv["window_len"]
                stride = loader_rv["stride"]
                image_size = config.get("image_size")

                logger.info(
                    f"{window_len=}, {stride=}, resolution={image_size}x{image_size}"
                )

                n_workers = config.get("n_workers")

                train_ts2i_start = perf_counter()
                number_of_train_images = create_png_image_dataset(
                    X=X_train,
                    y=y_train,
                    window_len=window_len,
                    stride=stride,
                    image_size=image_size,
                    ts2i_transformation_fn=ts2i_transformation,
                    output_dir=train_dir,
                    train=True,
                    n_workers=n_workers,
                )
                train_ts2i_stop = perf_counter()
                train_ts2i_transformation_time_s = train_ts2i_stop - train_ts2i_start

                test_ts2i_start = perf_counter()
                number_of_test_images = create_png_image_dataset(
                    X=X_test,
                    y=y_test,
                    window_len=window_len,
                    stride=stride,
                    image_size=image_size,
                    ts2i_transformation_fn=ts2i_transformation,
                    output_dir=test_dir,
                    train=False,
                    n_workers=n_workers,
                )
                test_ts2i_stop = perf_counter()
                test_ts2i_transformation_time_s = test_ts2i_stop - test_ts2i_start

                metadata = {
                    "file_path": _file_path,
                    "dataset": loader_rv["dataset"],
                    "config": OmegaConf.to_container(config, resolve=True),
                    "device": device,
                    "slidingWindow_TSB_AD_evaluation": loader_rv[
                        "slidingWindow_TSB_AD_evaluation"
                    ],
                    "window_len": window_len,
                    "stride": stride,
                    "ts2i_transformation_name": ts2i_transformation.__name__,
                    "ts2i_transformation_pretty_name": ts2i_transformation.pretty_name,
                    "number_of_train_images": number_of_train_images,
                    "number_of_test_images": number_of_test_images,
                    "n_workers": n_workers,
                    "theoretical_speedup": n_workers if n_workers > 0 else 1,
                    # Wall time (real execution time)
                    "train_ts2i_transformation_wall_time_s": train_ts2i_transformation_time_s,
                    "test_ts2i_transformation_wall_time_s": test_ts2i_transformation_time_s,
                    # Approximate CPU time when sequential execution
                    "train_ts2i_transformation_estimate_sequential_time_s": train_ts2i_transformation_time_s
                    * n_workers,
                    "test_ts2i_transformation_estimate_sequential_time_s": test_ts2i_transformation_time_s
                    * n_workers,
                    # Throughput
                    "train_ts2i_transformation_throughput_images_per_sec": number_of_train_images
                    / train_ts2i_transformation_time_s
                    if train_ts2i_transformation_time_s > 0
                    else 0,
                    "test_ts2i_transformation_throughput_images_per_sec": number_of_test_images
                    / test_ts2i_transformation_time_s
                    if test_ts2i_transformation_time_s > 0
                    else 0,
                    # Transformation time per one image
                    "train_ts2i_transformation_time_per_image_ms": (
                        train_ts2i_transformation_time_s / number_of_train_images * 1000
                    )
                    if number_of_train_images > 0
                    else 0,
                    "test_ts2i_transformation_time_per_image_ms": (
                        test_ts2i_transformation_time_s / number_of_test_images * 1000
                    )
                    if number_of_test_images > 0
                    else 0,
                    "timestamp": datetime.now().isoformat(),
                    "test_timeseries_length": len(X_test),
                    "y_test": y_test.tolist(),
                }

                with open(tmp_base / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                dataset_dir = Path(output_dir) / loader_rv["dataset"]
                dataset_dir.mkdir(parents=True, exist_ok=True)

                zip_name = f"{file_path.stem}_ts2i_{ts2i_transformation.pretty_name}"
                zip_path = dataset_dir / zip_name

                # Save metadata outside ZIP as well
                metadata_path = dataset_dir / f"{zip_name}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Saved metadata to {metadata_path}")

                shutil.make_archive(str(zip_path), "zip", tmp_base)
                logger.info(f"Saved to {zip_path.with_suffix('.zip')}")
