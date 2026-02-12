from src.transformations.typing import Array1D, ArrayImage
from src.transformations.ts2i import named
import numpy as np


@named("SPIRAL")
def spiral(window: Array1D, size: int) -> ArrayImage:
    window = np.ravel(window)
    target_size = size
    n = len(window)

    # Don't normalize as window is already scaled with MinMaxScaler
    norm_window = window

    center = target_size / 2

    # Create meshgrid of coordinates (alternative to nested for loops)
    y_coords, x_coords = np.ogrid[:target_size, :target_size]

    dx = x_coords - center
    dy = y_coords - center
    r = np.sqrt(dx * dx + dy * dy)

    max_r = target_size / 2
    mask = r <= max_r

    theta = np.arctan2(dy, dx)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)

    # Archimedean spiral mapping
    norm_r = r / max_r
    arms = 2
    spiral_theta = theta + norm_r * 2 * np.pi * arms

    # Map to point index
    point_idx = ((spiral_theta / (2 * np.pi)) * n / arms).astype(int) % n

    image = np.zeros((target_size, target_size), dtype=np.float32)

    image[mask] = norm_window[point_idx[mask]]

    return np.stack([image, image, image], axis=0)
