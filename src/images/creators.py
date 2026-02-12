from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import numpy as np
from PIL import Image
import pathlib
from typing import Callable
import os
from src.utils.logging import logger


def _process_single_window_shm(
    i: int,
    shm_X_name: str,
    shm_y_name: str,
    X_shape: tuple,
    X_dtype: np.dtype,
    y_shape: tuple,
    y_dtype: np.dtype,
    window_len: int,
    image_size: int,
    ts2i_transformation_fn: Callable,
    output_dir: pathlib.Path,
    train: bool,
    **kwargs,
):
    # Attach to existing shared memory
    shm_X = shared_memory.SharedMemory(name=shm_X_name)
    shm_y = shared_memory.SharedMemory(name=shm_y_name)

    # Create numpy arrays from shared memory
    X = np.ndarray(X_shape, dtype=X_dtype, buffer=shm_X.buf)
    y = np.ndarray(y_shape, dtype=y_dtype, buffer=shm_y.buf)

    try:
        sample = ts2i_transformation_fn(
            X[i : i + window_len].astype(np.float32), image_size, **kwargs
        )

        # Convert from (C,H,W) to (H,W,C)
        sample = np.transpose(sample, (1, 2, 0))
        sample_uint8 = (sample * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(sample_uint8)

        # label = 1 if ANY anomaly in window
        label = 1 if y[i : i + window_len].max() > 0 else 0

        # Semi-supervised settings
        if train and label == 1:
            return None

        image.save(output_dir / str(label) / f"sample_{i:09d}.png")
        return 1
    finally:
        # Close (but don't unlink) shared memory in worker
        shm_X.close()
        shm_y.close()


def create_png_image_dataset(
    X: np.ndarray,
    y: np.ndarray,
    window_len: int,
    stride: int,
    image_size: int,
    ts2i_transformation_fn: Callable,
    output_dir: pathlib.Path,
    train: bool,
    n_workers: int = None,
    **kwargs,
):
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    logger.info(f"Running with n_workers: {n_workers} for train={train} set")

    indices = list(range(0, len(X) - window_len, stride))

    # For small datasets, don't use multiprocessing
    if len(indices) < 100 or n_workers == 1:
        counter = 0
        for i in indices:
            sample = ts2i_transformation_fn(
                X[i : i + window_len].astype(np.float32), image_size, **kwargs
            )
            sample = np.transpose(sample, (1, 2, 0))
            sample_uint8 = (sample * 255).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(sample_uint8)

            label = 1 if y[i : i + window_len].max() > 0 else 0

            if train and label == 1:
                continue

            image.save(output_dir / str(label) / f"sample_{i:09d}.png")
            counter += 1
        return counter

    # Create shared memory for X and y
    shm_X = shared_memory.SharedMemory(create=True, size=X.nbytes)
    shm_y = shared_memory.SharedMemory(create=True, size=y.nbytes)

    try:
        # Copy data to shared memory
        shared_X = np.ndarray(X.shape, dtype=X.dtype, buffer=shm_X.buf)
        shared_y = np.ndarray(y.shape, dtype=y.dtype, buffer=shm_y.buf)
        np.copyto(shared_X, X)
        np.copyto(shared_y, y)

        # Parallel processing
        counter = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in indices:
                future = executor.submit(
                    _process_single_window_shm,
                    i=i,
                    shm_X_name=shm_X.name,
                    shm_y_name=shm_y.name,
                    X_shape=X.shape,
                    X_dtype=X.dtype,
                    y_shape=y.shape,
                    y_dtype=y.dtype,
                    window_len=window_len,
                    image_size=image_size,
                    ts2i_transformation_fn=ts2i_transformation_fn,
                    output_dir=output_dir,
                    train=train,
                    **kwargs,
                )
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    counter += 1

        return counter

    finally:
        # Cleanup shared memory
        shm_X.close()
        shm_X.unlink()
        shm_y.close()
        shm_y.unlink()
