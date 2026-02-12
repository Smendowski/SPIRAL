from src.transformations.typing import Array1D, ArrayImage
import numpy as np
from src.transformations.ts2i import named


@named("RN")
def random_noise(window: Array1D, size: int) -> ArrayImage:
    img = np.random.randn(size, size)
    return np.stack([img, img, img], axis=0).astype(np.float32)
